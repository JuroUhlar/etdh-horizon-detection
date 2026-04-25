"""
Attempt 10 - multi-cue candidates + temporal prior + top-connected sky envelope.

Motivation
----------
Attempts 4 and 6 share a single-channel Otsu classifier (gray for #4, dual
gray+b* with confidence selection for #6). Both fail the same way: when the
Otsu mask latches onto the wrong feature (overcast sky ≈ ground luminance,
sun glare, haze band), RANSAC fits a high-inlier line to a non-horizon
boundary. Inlier count is a poor proxy for "is this the real horizon".

Attempt 6's b* channel is a useful second hypothesis source, but selecting
a winner by inlier count was wrong: the channel that hallucinates a stronger
edge wins even when its line is geometrically off.

What's new relative to attempt 9
--------------------------------
1. Boundary points are extracted from BOTH grayscale and Lab b*. Each channel
   gets its own RANSAC top-K hypotheses; all candidates are pooled.

2. A low-resolution top-vs-bottom colour likelihood model adds one smooth
   dynamic-programming boundary candidate. This targets frames where Otsu
   thresholds omit the true horizon from both channel boundary pools.

3. A second low-resolution candidate looks for sky-like pixels connected to the
   top image border, with texture as an extra cue. The lower envelope of that
   top-connected component can recover a horizon when the strongest colour split
   is a treetop/field boundary lower in the frame.

4. Pooled candidates are re-ranked with an Ettinger-style region-coherence
   score on a downsampled Lab thumbnail: ||mu_above - mu_below||^2 divided
   by the within-region scatter (trace of pooled covariance). This directly
   asks "does this line cleanly separate two color-coherent regions?", which
   is what the horizon physically does — independent of which channel made
   the candidate.

5. When adjacent frames look like the same video scene, candidates close to
   the previous accepted line get a soft temporal boost before reranking.
   This targets the FPV/ATV failure mass where single-frame masks jump to
   tree/field edges while the real horizon moves more smoothly.

6. The winning candidate's inliers are refit with Huber (cv2.fitLine) for
   sub-pixel angle and offset.

7. The detector abstains ("no_horizon") when:
   - both channels produced a near-degenerate mask (>=92% one class), OR
   - best Ettinger coherence score is below FALLBACK_COHERENCE.
   This addresses the 10 no-horizon frames in data/video_clips_fpv_atv that
   every previous attempt fits a line to and gets a guaranteed pass-fail on.

Pipeline cost (480x480 frames)
------------------------------
~10-20 ms on dev host:
  - Lab + gray conversion:      ~1 ms
  - Per-channel boundary mask:  ~1 ms x 2
  - Vectorised RANSAC top-K:    ~1 ms x 2
  - Ettinger rerank (60x60):    ~1 ms
  - Huber fit + mask emission:  ~1 ms
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

_MAX_BOUNDARY_PTS    = 400    # subsample cap before RANSAC (per channel)
_RANSAC_ITER         = 200    # iterations per channel
_TOP_K               = 8      # top hypotheses kept per channel
_INLIER_THRESHOLD    = 3.0    # px, RANSAC inlier band
_RERANK_THUMB        = 60     # side length of Lab thumbnail used for rerank
_DEGENERATE_FRACTION = 0.92   # mask "no horizon" if >= this fraction is one class on both channels
_FALLBACK_COHERENCE  = 0.22   # best Ettinger score below this -> abstain
_MIN_BOUNDARY_PTS    = 8      # below this, no boundary -> abstain

_SCENE_THUMB_SIZE          = 16
_SCENE_CHANGE_MAD          = 22.0   # 16x16 Lab-L mean absolute diff; reset temporal state above this
_TEMPORAL_THETA_SIGMA_DEG  = 8.0
_TEMPORAL_RHO_SIGMA_NORM   = 0.14
_TEMPORAL_WEIGHT           = 2.0

_DP_THUMB_W               = 96
_DP_THUMB_H               = 72
_DP_TOP_SEED_FRAC         = 0.18
_DP_BOTTOM_SEED_FRAC      = 0.25
_DP_MAX_STEP              = 3
_DP_STEP_PENALTY          = 0.35
_DP_MIN_SEED_SEPARATION   = 0.38
_DP_MIN_PATH_MARGIN       = 0.18
_DP_MIN_RERANK_COHERENCE  = 5.0
_DP_MAX_ROLL_DEG          = 25.0

_SKY_THUMB_W                  = 96
_SKY_THUMB_H                  = 72
_SKY_TOP_SEED_FRAC            = 0.18
_SKY_BOTTOM_SEED_FRAC         = 0.25
_SKY_MIN_SEED_SEPARATION      = 0.24
_SKY_LLR_THRESHOLD            = 0.35
_SKY_MIN_AREA_FRAC            = 0.08
_SKY_MAX_AREA_FRAC            = 0.72
_SKY_MIN_COLUMN_COVERAGE      = 0.58
_SKY_MIN_MEDIAN_BOUNDARY_FRAC = 0.10
_SKY_MAX_MEDIAN_BOUNDARY_FRAC = 0.86
_SKY_MIN_TEXTURE_CONTRAST     = 0.035
_SKY_MIN_RERANK_COHERENCE     = 1.00
_SKY_MAX_ROLL_DEG             = 65.0
_SKY_SCORE_BOOST              = 1.20
_SKY_OVERRIDE_SCORE_RATIO     = 0.72
_SKY_OVERRIDE_MIN_DELTA_FRAC  = 0.08

# Boundary orientation filter: tree trunks, building edges and frame borders
# create long, collinear boundary segments that RANSAC happily fits. The real
# horizon's mask gradient is near-vertical (gy >> gx) up to the maximum
# expected roll. We discard boundary pixels whose local gradient is closer
# to horizontal than `90° - _MAX_ROLL_DEG` from horizontal, which physically
# means "boundary tangent steeper than _MAX_ROLL_DEG". This keeps all
# legitimate roll candidates while removing vertical-edge clutter.
_MAX_ROLL_DEG = 75.0

# Angle prior: real horizons are typically near-horizontal in aerial/FPV
# footage. Lines steeper than ~60° usually correspond to vertical structure
# (tree trunks, building edges, frame borders). The dataset README confirms
# Horizon-UAV roll is bounded at ~|57°|; FPV/ATV labels stay below ~|45°|.
# Below the soft band the score is left untouched; above it the penalty
# grows quickly so a near-vertical candidate must have overwhelmingly
# better coherence to win.
_ANGLE_SOFT_DEG = 60.0   # full strength below this
_ANGLE_HARD_DEG = 85.0   # near-zero weight at and beyond this

_rng = np.random.default_rng()
_prev_line: tuple[float, float, float, float] | None = None
_prev_scene_thumb: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Channel-specific boundary extraction (same morph recipe as attempts 2-6)
# ---------------------------------------------------------------------------

def _extract_boundary(channel_image: np.ndarray, max_roll_deg: float = _MAX_ROLL_DEG):
    """Otsu + morph close/open + gradient + orientation filter.

    The orientation filter discards boundary pixels whose local gradient is
    nearly horizontal (i.e. the boundary itself is nearly vertical). This is
    what removes tree trunks, building edges and frame borders from RANSAC
    consideration without rejecting genuinely rolled horizons up to
    `max_roll_deg`. If the filter would leave too few pixels we fall back to
    the unfiltered boundary on that channel.

    Returns (boundary_mask, sky_ground_mask, mask_balance) where mask_balance
    is the larger-class fraction of mask (used for degeneracy abstention).
    """
    blurred = cv2.GaussianBlur(channel_image, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    boundary_raw = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary_raw[0, :] = boundary_raw[-1, :] = boundary_raw[:, 0] = boundary_raw[:, -1] = 0

    boundary = boundary_raw
    if max_roll_deg < 90.0 and int(np.count_nonzero(boundary_raw)) > _MIN_BOUNDARY_PTS:
        # Sobel on the float mask -> signed gradient at every pixel.
        mask_f = mask.astype(np.float32)
        gx = cv2.Sobel(mask_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(mask_f, cv2.CV_32F, 0, 1, ksize=3)

        # We want |gy| / sqrt(gx^2 + gy^2) > sin(90 - max_roll_deg).
        # Squared form avoids the sqrt and matches gy*gy/(gx^2 + gy^2).
        threshold_sin = np.sin(np.deg2rad(90.0 - max_roll_deg))
        keep = (gy * gy) > (threshold_sin * threshold_sin) * (gx * gx + gy * gy + 1e-6)
        filtered = np.where(keep, boundary_raw, 0).astype(np.uint8)
        filtered[0, :] = filtered[-1, :] = filtered[:, 0] = filtered[:, -1] = 0

        # Only adopt the filtered boundary if it preserved enough points.
        if int(np.count_nonzero(filtered)) >= _MIN_BOUNDARY_PTS:
            boundary = filtered

    n = mask.size
    n_white = int(np.count_nonzero(mask))
    balance = max(n_white, n - n_white) / n   # 0.5 = perfectly split, 1.0 = degenerate

    return boundary, mask, balance


# ---------------------------------------------------------------------------
# Top-K vectorised RANSAC
# ---------------------------------------------------------------------------

def _ransac_topk(
    points: np.ndarray,
    n_iter: int,
    inlier_threshold: float,
    k: int,
    rng: np.random.Generator,
) -> list[tuple[int, np.ndarray, tuple[float, float, float, float]]]:
    """Return up to k hypotheses (count, inlier_mask, line_seed) sorted desc.

    line_seed = (vx, vy, x0, y0): the seed pair used so we can recover the
    direction without re-solving from the inliers (we'll Huber-refit later).
    """
    n_pts = len(points)
    if n_pts < 2:
        return []

    idx_a = rng.integers(0, n_pts, size=n_iter)
    idx_b = rng.integers(0, n_pts - 1, size=n_iter)
    idx_b += (idx_b >= idx_a).astype(idx_b.dtype)

    pa = points[idx_a]    # (n_iter, 2)
    pb = points[idx_b]

    delta  = (pb - pa).astype(np.float32)
    length = np.hypot(delta[:, 0], delta[:, 1])
    valid  = length > 1e-6
    safe   = np.where(valid, length, 1.0)
    vx = delta[:, 0] / safe
    vy = delta[:, 1] / safe

    # Orthogonal distance from every point to every hypothesis: (n_iter, n_pts)
    diff_x = points[:, 0][None, :] - pa[:, 0:1]
    diff_y = points[:, 1][None, :] - pa[:, 1:2]
    dist   = np.abs(diff_x * vy[:, None] - diff_y * vx[:, None])

    inlier_masks  = dist < inlier_threshold
    inlier_counts = inlier_masks.sum(axis=1).astype(np.int32)
    inlier_counts[~valid] = 0

    if int(inlier_counts.max()) < 2:
        return []

    k = min(k, n_iter)
    top_idx = np.argpartition(-inlier_counts, k - 1)[:k]
    top_idx = top_idx[np.argsort(-inlier_counts[top_idx])]

    out: list[tuple[int, np.ndarray, tuple[float, float, float, float]]] = []
    for i in top_idx:
        c = int(inlier_counts[i])
        if c < 2:
            continue
        out.append((
            c,
            inlier_masks[i].copy(),
            (float(vx[i]), float(vy[i]), float(pa[i, 0]), float(pa[i, 1])),
        ))
    return out


# ---------------------------------------------------------------------------
# Low-resolution colour-likelihood dynamic-programming boundary
# ---------------------------------------------------------------------------

def _robust_location_scale(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median), axis=0)
    scale = np.maximum(1.4826 * mad, 6.0)
    return median.astype(np.float32), scale.astype(np.float32)


def _dp_features(image_bgr: np.ndarray, lab: np.ndarray) -> np.ndarray:
    small_bgr = cv2.resize(
        image_bgr,
        (_DP_THUMB_W, _DP_THUMB_H),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    small_lab = cv2.resize(
        lab,
        (_DP_THUMB_W, _DP_THUMB_H),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    small_hsv = cv2.cvtColor(small_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    blue = small_bgr[:, :, 0]
    red = small_bgr[:, :, 2]
    blue_excess = 128.0 + 128.0 * (blue - red) / (blue + red + 1.0)

    return np.dstack(
        [
            small_lab[:, :, 0],
            small_lab[:, :, 1],
            small_lab[:, :, 2],
            small_hsv[:, :, 1],
            blue_excess,
        ]
    ).astype(np.float32)


def _diag_log_likelihood(features: np.ndarray, mu: np.ndarray, scale: np.ndarray) -> np.ndarray:
    z = (features - mu) / scale
    return -0.5 * np.sum(z * z + 2.0 * np.log(scale), axis=2)


def _best_dp_path(column_scores: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = column_scores.shape
    dp = column_scores[:, 0].copy()
    back = np.zeros((w, h), dtype=np.int16)
    rows = np.arange(h, dtype=np.int16)
    deltas = np.arange(-_DP_MAX_STEP, _DP_MAX_STEP + 1, dtype=np.int16)

    for x in range(1, w):
        candidates = np.full((len(deltas), h), -np.inf, dtype=np.float32)
        for i, delta in enumerate(deltas):
            penalty = _DP_STEP_PENALTY * abs(int(delta))
            if delta < 0:
                candidates[i, -delta:] = dp[: h + delta] - penalty
            elif delta > 0:
                candidates[i, : h - delta] = dp[delta:] - penalty
            else:
                candidates[i] = dp
        best_delta_idx = np.argmax(candidates, axis=0)
        dp = column_scores[:, x] + candidates[best_delta_idx, rows]
        back[x] = rows + deltas[best_delta_idx]

    path = np.empty(w, dtype=np.int16)
    path[-1] = int(np.argmax(dp))
    best_score = float(dp[path[-1]])
    for x in range(w - 1, 0, -1):
        path[x - 1] = back[x, path[x]]
    return path, best_score


def _path_mask(path: np.ndarray, full_w: int, full_h: int) -> np.ndarray:
    low_mask = np.zeros((_DP_THUMB_H, _DP_THUMB_W), dtype=np.uint8)
    for x, y in enumerate(path):
        low_mask[: int(y), x] = 255
    return cv2.resize(low_mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)


def _likelihood_dp_candidate(
    image_bgr: np.ndarray,
    lab: np.ndarray,
    full_w: int,
    full_h: int,
) -> tuple[int, np.ndarray, tuple[float, float, float, float], np.ndarray, np.ndarray] | None:
    """Return one smooth boundary candidate from top/bottom colour models.

    The model assumes sky evidence is most likely in the upper band and ground
    evidence in the lower band. It is intentionally conservative: if the seed
    bands are not distinguishable or the path objective is weak, no candidate is
    added to the pooled reranker.
    """
    features = _dp_features(image_bgr, lab)
    top_rows = max(4, int(round(_DP_THUMB_H * _DP_TOP_SEED_FRAC)))
    bottom_rows = max(4, int(round(_DP_THUMB_H * _DP_BOTTOM_SEED_FRAC)))

    sky_samples = features[:top_rows].reshape(-1, features.shape[2])
    ground_samples = features[-bottom_rows:].reshape(-1, features.shape[2])
    sky_mu, sky_scale = _robust_location_scale(sky_samples)
    ground_mu, ground_scale = _robust_location_scale(ground_samples)

    pooled_scale = sky_scale + ground_scale
    seed_separation = float(np.sqrt(np.mean(((sky_mu - ground_mu) / pooled_scale) ** 2)))
    if seed_separation < _DP_MIN_SEED_SEPARATION:
        return None

    sky_ll = _diag_log_likelihood(features, sky_mu, sky_scale)
    ground_ll = _diag_log_likelihood(features, ground_mu, ground_scale)
    llr = np.clip(sky_ll - ground_ll, -8.0, 8.0).astype(np.float32)

    prefix = np.vstack([
        np.zeros((1, _DP_THUMB_W), dtype=np.float32),
        np.cumsum(llr, axis=0),
    ])
    total = prefix[-1]
    split_scores = (2.0 * prefix - total[None, :]).astype(np.float32)
    split_scores[:3, :] -= 2.0
    split_scores[-3:, :] -= 2.0

    path_with_edges, best_score = _best_dp_path(split_scores)
    path = np.clip(path_with_edges, 0, _DP_THUMB_H - 1)
    path_margin = best_score / float(_DP_THUMB_W * _DP_THUMB_H)
    if path_margin < _DP_MIN_PATH_MARGIN:
        return None

    xs = (np.arange(_DP_THUMB_W, dtype=np.float32) + 0.5) * (full_w / _DP_THUMB_W)
    ys = (path.astype(np.float32) + 0.5) * (full_h / _DP_THUMB_H)
    points = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    line = (float(vx), float(vy), float(x0), float(y0))

    confidence = min(1.0, path_margin / 1.5)
    count = max(2, int(round(confidence * len(points))))
    inlier_mask = np.ones(len(points), dtype=bool)
    return count, inlier_mask, line, points, _path_mask(path, full_w, full_h)


# ---------------------------------------------------------------------------
# Low-resolution top-connected sky envelope candidate
# ---------------------------------------------------------------------------

def _sky_envelope_features(
    image_bgr: np.ndarray,
    lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    small_bgr = cv2.resize(
        image_bgr,
        (_SKY_THUMB_W, _SKY_THUMB_H),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    small_lab = cv2.resize(
        lab,
        (_SKY_THUMB_W, _SKY_THUMB_H),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    small_hsv = cv2.cvtColor(small_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)

    luma = small_lab[:, :, 0]
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.abs(gx) + np.abs(gy)
    grad_scale = max(float(np.percentile(grad, 95)), 1.0)
    texture = cv2.boxFilter(
        np.clip(grad / grad_scale, 0.0, 1.0),
        ddepth=-1,
        ksize=(5, 5),
        normalize=True,
    ).astype(np.float32)

    blue = small_bgr[:, :, 0]
    red = small_bgr[:, :, 2]
    blue_excess = 128.0 + 128.0 * (blue - red) / (blue + red + 1.0)

    features = np.dstack(
        [
            small_lab[:, :, 0],
            small_lab[:, :, 1],
            small_lab[:, :, 2],
            small_hsv[:, :, 1],
            blue_excess,
            96.0 * texture,
        ]
    ).astype(np.float32)
    return features, texture


def _top_connected_mask(sky_like: np.ndarray) -> np.ndarray:
    n_labels, labels = cv2.connectedComponents(sky_like, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(sky_like, dtype=bool)

    top_labels = np.unique(labels[:2, :])
    top_labels = top_labels[top_labels != 0]
    if len(top_labels) == 0:
        return np.zeros_like(sky_like, dtype=bool)
    return np.isin(labels, top_labels)


def _envelope_mask(path: np.ndarray, valid_columns: np.ndarray, full_w: int, full_h: int) -> np.ndarray:
    low_mask = np.zeros((_SKY_THUMB_H, _SKY_THUMB_W), dtype=np.uint8)
    for x, y in enumerate(path):
        if valid_columns[x]:
            low_mask[: int(y) + 1, x] = 255
    return cv2.resize(low_mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)


def _top_connected_sky_candidate(
    image_bgr: np.ndarray,
    lab: np.ndarray,
    full_w: int,
    full_h: int,
) -> tuple[int, np.ndarray, tuple[float, float, float, float], np.ndarray, np.ndarray] | None:
    """Fit a line to the lower envelope of a top-connected sky-like component."""
    features, texture = _sky_envelope_features(image_bgr, lab)
    top_rows = max(4, int(round(_SKY_THUMB_H * _SKY_TOP_SEED_FRAC)))
    bottom_rows = max(4, int(round(_SKY_THUMB_H * _SKY_BOTTOM_SEED_FRAC)))

    sky_samples = features[:top_rows].reshape(-1, features.shape[2])
    ground_samples = features[-bottom_rows:].reshape(-1, features.shape[2])
    sky_mu, sky_scale = _robust_location_scale(sky_samples)
    ground_mu, ground_scale = _robust_location_scale(ground_samples)

    pooled_scale = sky_scale + ground_scale
    seed_separation = float(np.sqrt(np.mean(((sky_mu - ground_mu) / pooled_scale) ** 2)))
    if seed_separation < _SKY_MIN_SEED_SEPARATION:
        return None

    sky_ll = _diag_log_likelihood(features, sky_mu, sky_scale)
    ground_ll = _diag_log_likelihood(features, ground_mu, ground_scale)
    llr = np.clip(sky_ll - ground_ll, -8.0, 8.0).astype(np.float32)
    sky_like = (llr > _SKY_LLR_THRESHOLD).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    sky_like = cv2.morphologyEx(sky_like, cv2.MORPH_CLOSE, kernel)
    sky_like = cv2.morphologyEx(sky_like, cv2.MORPH_OPEN, kernel)

    sky_mask = _top_connected_mask(sky_like)
    area_frac = float(sky_mask.mean())
    if area_frac < _SKY_MIN_AREA_FRAC or area_frac > _SKY_MAX_AREA_FRAC:
        return None

    valid_columns = sky_mask.any(axis=0)
    column_coverage = float(valid_columns.mean())
    if column_coverage < _SKY_MIN_COLUMN_COVERAGE:
        return None

    path = np.zeros(_SKY_THUMB_W, dtype=np.int16)
    for x in np.flatnonzero(valid_columns):
        path[x] = int(np.flatnonzero(sky_mask[:, x])[-1])

    valid_x = np.flatnonzero(valid_columns)
    valid_y = path[valid_columns].astype(np.float32)
    median_boundary_frac = float(np.median(valid_y) / max(_SKY_THUMB_H - 1, 1))
    if (
        median_boundary_frac < _SKY_MIN_MEDIAN_BOUNDARY_FRAC
        or median_boundary_frac > _SKY_MAX_MEDIAN_BOUNDARY_FRAC
    ):
        return None

    below_mask = np.zeros_like(sky_mask, dtype=bool)
    for x in valid_x:
        below_mask[int(path[x]) + 1 :, x] = True
    if int(below_mask.sum()) < 16:
        return None

    texture_contrast = float(texture[below_mask].mean() - texture[sky_mask].mean())
    if texture_contrast < _SKY_MIN_TEXTURE_CONTRAST:
        return None

    xs = (valid_x.astype(np.float32) + 0.5) * (full_w / _SKY_THUMB_W)
    ys = (valid_y + 0.5) * (full_h / _SKY_THUMB_H)
    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) < _MIN_BOUNDARY_PTS:
        return None

    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    line = (float(vx), float(vy), float(x0), float(y0))

    confidence = min(1.0, column_coverage * (1.0 + 4.0 * texture_contrast))
    count = max(2, int(round(confidence * len(points))))
    inlier_mask = np.ones(len(points), dtype=bool)
    mask = _envelope_mask(path, valid_columns, full_w, full_h)
    return count, inlier_mask, line, points, mask


# ---------------------------------------------------------------------------
# Ettinger-style region coherence rerank
# ---------------------------------------------------------------------------

def _angle_prior(vx: float, vy: float) -> float:
    """Soft prior on line orientation; favours near-horizontal candidates.

    Returns a multiplicative weight in (~0.05, 1.0]. Angles within
    _ANGLE_SOFT_DEG of horizontal get full weight; the weight then drops
    smoothly to ~0.05 at _ANGLE_HARD_DEG and stays there. This penalises
    vertical tree trunks and building edges without rejecting genuinely
    rolled horizons (Horizon-UAV reaches ~57°).
    """
    angle = abs(np.degrees(np.arctan2(vy, vx)))
    if angle > 90.0:
        angle = 180.0 - angle
    if angle <= _ANGLE_SOFT_DEG:
        return 1.0
    if angle >= _ANGLE_HARD_DEG:
        return 0.05
    t = (angle - _ANGLE_SOFT_DEG) / (_ANGLE_HARD_DEG - _ANGLE_SOFT_DEG)
    return float(1.0 - 0.95 * t)


def _hesse(line: tuple[float, float, float, float]) -> tuple[float, float, float]:
    vx, vy, x0, y0 = line
    length = float(np.hypot(vx, vy))
    nx, ny = -vy / length, vx / length
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny
    return nx, ny, nx * x0 + ny * y0


def _angular_delta_deg(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    angle_a = float(np.degrees(np.arctan2(a[1], a[0])))
    angle_b = float(np.degrees(np.arctan2(b[1], b[0])))
    raw = abs(angle_a - angle_b) % 180.0
    return min(raw, 180.0 - raw)


def _temporal_prior(
    line: tuple[float, float, float, float],
    prev_line: tuple[float, float, float, float] | None,
    image_height: int,
) -> float:
    if prev_line is None:
        return 0.0
    d_theta = _angular_delta_deg(line, prev_line)
    _, _, rho = _hesse(line)
    _, _, prev_rho = _hesse(prev_line)
    d_rho_norm = abs(rho - prev_rho) / max(float(image_height), 1.0)
    theta_term = d_theta / _TEMPORAL_THETA_SIGMA_DEG
    rho_term = d_rho_norm / _TEMPORAL_RHO_SIGMA_NORM
    return float(np.exp(-(theta_term * theta_term + rho_term * rho_term)))


def _roll_deg(line: tuple[float, float, float, float]) -> float:
    vx, vy, _x0, _y0 = line
    roll = abs(float(np.degrees(np.arctan2(vy, vx))))
    if roll > 90.0:
        roll = 180.0 - roll
    return roll


def _line_y_at_x(line: tuple[float, float, float, float], x: float) -> float:
    vx, vy, x0, y0 = line
    if abs(vx) < 1e-6:
        return float(y0)
    return float(y0 + (vy / vx) * (x - x0))


def _scene_thumb(gray: np.ndarray) -> np.ndarray:
    return cv2.resize(
        gray,
        (_SCENE_THUMB_SIZE, _SCENE_THUMB_SIZE),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def _scene_is_continuous(scene_thumb: np.ndarray) -> bool:
    if _prev_scene_thumb is None:
        return False
    diff = float(np.mean(np.abs(scene_thumb - _prev_scene_thumb)))
    return diff <= _SCENE_CHANGE_MAD


def _reset_temporal_state(scene_thumb: np.ndarray | None = None) -> None:
    global _prev_line, _prev_scene_thumb
    _prev_line = None
    _prev_scene_thumb = scene_thumb


def _remember_temporal_line(
    line: tuple[float, float, float, float],
    scene_thumb: np.ndarray,
) -> None:
    global _prev_line, _prev_scene_thumb
    _prev_line = line
    _prev_scene_thumb = scene_thumb


def _ettinger_score(
    line: tuple[float, float, float, float],
    lab_thumb: np.ndarray,
    full_w: int,
    full_h: int,
) -> float:
    """Score how cleanly `line` separates lab_thumb into two coherent regions.

    Score = (||mu_above - mu_below||^2 / (trace(cov_above) + trace(cov_below) + eps))
            * angle_prior(line)

    Lab pixels are used because they expose both the luminance contrast that
    grayscale Otsu sees AND the blue/yellow contrast that the b* channel sees,
    in the same metric. The angle prior multiplies the geometric coherence
    score so a near-vertical candidate must beat horizontal candidates by a
    wide margin to win. Higher = better.

    Note: a `texture_asymmetry = 1 + |var_a - var_b| / (var_a + var_b)`
    multiplier was tried (intent: reward smooth-vs-textured splits) and
    measurably regressed both datasets (-0.8 pp on UAV, -3.3 pp on FPV).
    See `result.md` for the failed-experiment notes.
    """
    vx, vy, x0, y0 = line
    sh, sw = lab_thumb.shape[:2]

    sx0 = x0 * sw / full_w
    sy0 = y0 * sh / full_h

    ys, xs = np.mgrid[0:sh, 0:sw]
    side = (xs - sx0) * (-vy) + (ys - sy0) * vx
    above_mask = side > 0

    n_above = int(above_mask.sum())
    n_below = sh * sw - n_above
    if n_above < 16 or n_below < 16:
        return -1.0

    pixels = lab_thumb.reshape(-1, 3).astype(np.float32)
    above_pix = pixels[above_mask.flatten()]
    below_pix = pixels[~above_mask.flatten()]

    mu_a = above_pix.mean(axis=0)
    mu_b = below_pix.mean(axis=0)

    var_a = above_pix.var(axis=0).sum()   # trace(cov_a)
    var_b = below_pix.var(axis=0).sum()

    diff = mu_a - mu_b
    between = float(diff @ diff)
    within  = float(var_a + var_b) + 1.0   # +1 keeps score finite on flat regions

    return (between / within) * _angle_prior(vx, vy)


# ---------------------------------------------------------------------------
# Huber refit -> result dict
# ---------------------------------------------------------------------------

def _build_result(
    inlier_pts: np.ndarray,
    confidence: float,
    coherence: float,
    mask: np.ndarray | None,
) -> dict:
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

    angle_deg = float(np.degrees(np.arctan2(vy, vx)))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg <= -90:
        angle_deg += 180

    intercept_y = float(y0 - (vy / vx) * x0) if abs(vx) > 1e-6 else float("nan")

    return {
        "angle_deg":         angle_deg,
        "intercept_y_at_x0": intercept_y,
        "line":              (float(vx), float(vy), float(x0), float(y0)),
        "confidence":        float(confidence),
        "inlier_count":      int(len(inlier_pts)),
        "coherence":         float(coherence),
        "mask":              mask,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_horizon(
    image_bgr: np.ndarray,
    ransac_iterations: int    = _RANSAC_ITER,
    inlier_threshold: float   = _INLIER_THRESHOLD,
    top_k: int                = _TOP_K,
    random_seed: "int | None" = None,
):
    """Estimate the horizon line, or return 'no_horizon' if none is plausible.

    Pipeline:
      1. Build per-channel boundary masks for grayscale and Lab b*.
      2. Run RANSAC top-K on each channel; pool candidates.
      3. Add low-resolution DP and top-connected sky candidates if confident.
      4. Rerank pooled candidates by Ettinger coherence on a Lab thumbnail.
      5. Refit the winner with Huber.
      6. Abstain if all signals say "no horizon present".

    Returns:
      - dict with keys angle_deg, intercept_y_at_x0, line, confidence,
        inlier_count, coherence, mask.
      - "no_horizon" string if the detector decides the frame has no horizon.
      - None on hard failure (boundary completely empty in both channels).
    """
    global _rng
    rng = np.random.default_rng(random_seed) if random_seed is not None else _rng

    h, w = image_bgr.shape[:2]

    lab  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
    gray = lab[:, :, 0]            # Lab L is just (a scaled) luminance — same role as cv2 BGR2GRAY
    b_star = lab[:, :, 2]
    scene_thumb = _scene_thumb(gray)
    use_temporal = _scene_is_continuous(scene_thumb)
    if not use_temporal:
        _reset_temporal_state(scene_thumb)

    boundaries: list[tuple[np.ndarray, np.ndarray, float]] = []
    for ch_img in (gray, b_star):
        boundary, mask, balance = _extract_boundary(ch_img)
        boundaries.append((boundary, mask, balance))

    # Degenerate mask check on both channels: if both are >=92% one class, the
    # frame is probably featureless sky-only or ground-only.
    if all(bal >= _DEGENERATE_FRACTION for _, _, bal in boundaries):
        _reset_temporal_state(scene_thumb)
        return "no_horizon"

    dp_entry = _likelihood_dp_candidate(image_bgr, lab, w, h)
    sky_entry = _top_connected_sky_candidate(image_bgr, lab, w, h)

    # Pool candidates from each channel.
    pooled: list[tuple[float, np.ndarray, tuple[float, float, float, float], np.ndarray, np.ndarray]] = []
    # tuple = (count, inlier_mask, line_seed, points_used, channel_mask_full)
    for boundary, mask, _balance in boundaries:
        ys, xs = np.where(boundary > 0)
        if len(xs) < _MIN_BOUNDARY_PTS:
            continue
        points = np.column_stack([xs, ys]).astype(np.float32)
        if len(points) > _MAX_BOUNDARY_PTS:
            idx    = rng.choice(len(points), _MAX_BOUNDARY_PTS, replace=False)
            points = points[idx]
        for count, inlier_mask, seed in _ransac_topk(points, ransac_iterations, inlier_threshold, top_k, rng):
            pooled.append((count, inlier_mask, seed, points, mask))

    # Rerank by Ettinger coherence on a Lab thumbnail. The same score is used
    # to admit the DP candidate so it only competes when it is a strong,
    # near-horizontal region split.
    thumb_size = (_RERANK_THUMB, _RERANK_THUMB)
    lab_thumb  = cv2.resize(lab, thumb_size, interpolation=cv2.INTER_AREA)

    if dp_entry is not None:
        dp_line = dp_entry[2]
        dp_roll = _roll_deg(dp_line)
        dp_score = _ettinger_score(dp_line, lab_thumb, w, h)
        if dp_score >= _DP_MIN_RERANK_COHERENCE and dp_roll <= _DP_MAX_ROLL_DEG:
            pooled.append(dp_entry)

    sky_raw_score = -float("inf")
    if sky_entry is not None:
        sky_line = sky_entry[2]
        sky_raw_score = _ettinger_score(sky_line, lab_thumb, w, h)
        if sky_raw_score >= _SKY_MIN_RERANK_COHERENCE and _roll_deg(sky_line) <= _SKY_MAX_ROLL_DEG:
            pooled.append(sky_entry)

    if not pooled:
        _reset_temporal_state(scene_thumb)
        return "no_horizon"

    best_score = -float("inf")
    best_raw_score = -float("inf")
    best_entry = None
    for entry in pooled:
        _count, _mask, seed, _pts, _full_mask = entry
        raw_score = _ettinger_score(seed, lab_thumb, w, h)
        score = raw_score
        if sky_entry is not None and entry is sky_entry:
            score *= _SKY_SCORE_BOOST
        if use_temporal:
            score *= 1.0 + _TEMPORAL_WEIGHT * _temporal_prior(seed, _prev_line, h)
        if score > best_score:
            best_score = score
            best_raw_score = raw_score
            best_entry = entry

    if sky_entry is not None and best_entry is not None and sky_entry is not best_entry:
        sky_line = sky_entry[2]
        best_line = best_entry[2]
        y_delta = _line_y_at_x(best_line, w * 0.5) - _line_y_at_x(sky_line, w * 0.5)
        if (
            sky_raw_score >= _SKY_MIN_RERANK_COHERENCE
            and sky_raw_score >= _SKY_OVERRIDE_SCORE_RATIO * best_raw_score
            and y_delta >= _SKY_OVERRIDE_MIN_DELTA_FRAC * h
        ):
            best_entry = sky_entry
            best_raw_score = sky_raw_score

    if best_entry is None or best_raw_score < _FALLBACK_COHERENCE:
        _reset_temporal_state(scene_thumb)
        return "no_horizon"

    count, inlier_mask, seed, points, full_mask = best_entry
    inlier_pts = points[inlier_mask].astype(np.float32)
    if len(inlier_pts) < 2:
        _reset_temporal_state(scene_thumb)
        return "no_horizon"

    confidence = count / max(len(points), 1)
    result = _build_result(inlier_pts, confidence, best_raw_score, full_mask)
    _remember_temporal_line(result["line"], scene_thumb)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _draw(image_bgr: np.ndarray, result) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    if result == "no_horizon" or result is None:
        cv2.putText(out, "no horizon", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return out
    vx, vy, x0, y0 = result["line"]
    scale = max(h, w) * 2
    p1 = (int(round(x0 - scale * vx)), int(round(y0 - scale * vy)))
    p2 = (int(round(x0 + scale * vx)), int(round(y0 + scale * vy)))
    cv2.line(out, p1, p2, (0, 0, 255), 2)
    offset = result["intercept_y_at_x0"]
    offset_str = "vertical" if np.isnan(offset) else f"{offset:+.1f}px"
    label = (
        f"angle={result['angle_deg']:+.2f}deg  offset={offset_str}  "
        f"conf={result['confidence']:.2f}  coh={result['coherence']:.2f}"
    )
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",        type=Path)
    parser.add_argument("--out",        type=Path,  default=None)
    parser.add_argument("--iterations", type=int,   default=_RANSAC_ITER)
    parser.add_argument("--threshold",  type=float, default=_INLIER_THRESHOLD)
    parser.add_argument("--seed",       type=int,   default=None)
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    t0 = time.perf_counter()
    result = detect_horizon(
        img,
        ransac_iterations=args.iterations,
        inlier_threshold=args.threshold,
        random_seed=args.seed,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if result is None:
        print(f"{args.image.name}: detection failed  ({elapsed_ms:.1f} ms)")
        return
    if result == "no_horizon":
        print(f"{args.image.name}: no horizon  ({elapsed_ms:.1f} ms)")
    else:
        offset_str = "vertical" if np.isnan(result["intercept_y_at_x0"]) else f"{result['intercept_y_at_x0']:+.1f}px"
        print(
            f"{args.image.name}: angle={result['angle_deg']:+.2f}deg  "
            f"offset={offset_str}  conf={result['confidence']:.2f}  "
            f"coherence={result['coherence']:.3f}  "
            f"inliers={result['inlier_count']}  ({elapsed_ms:.1f} ms)"
        )

    out_path = args.out or args.image.with_name(args.image.stem + "_horizon.jpg")
    cv2.imwrite(str(out_path), _draw(img, result))
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
