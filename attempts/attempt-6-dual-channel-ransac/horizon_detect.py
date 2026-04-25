"""
Attempt 6 — dual-channel RANSAC.

Changes from attempt 5:
  1. Dual classifier: the full boundary-extraction + RANSAC pipeline runs on
     BOTH the grayscale channel and the Lab b* channel; whichever produces
     the higher inlier confidence is returned.
     - Grayscale works well on horizon_uav and clear UAV footage.
     - Lab b* separates sky (blue → low b*) from vegetation/terrain
       (yellow/brown → high b*), helping where luminance contrast is low.
  2. Row-scan fallback: when both channels score below FALLBACK_CONF the row
     with maximum horizontal gradient energy is used as the horizon position
     at 0° roll, avoiding a random RANSAC line on truly ambiguous frames.

Carries over from attempt 5: vectorised RANSAC, batched early stopping,
no clustering (N=1), module-level RNG, boundary-point subsampling.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

_MAX_BOUNDARY_PTS = 400
_DEFAULT_ITER     = 300
_BATCH_SIZE       = 50
_EARLY_STOP_RATIO = 0.75
_INLIER_THRESHOLD = 3.0
_FALLBACK_CONF    = 0.30   # below this on both channels → row-scan fallback

_rng = np.random.default_rng()


# ---------------------------------------------------------------------------
# Boundary extraction — parameterised by channel
# ---------------------------------------------------------------------------

def _extract_boundary(image_bgr: np.ndarray, channel: str):
    """Return (boundary_mask, sky_ground_mask).

    channel = "gray"   — blurred grayscale (same as attempts 2–5)
    channel = "b_star" — Lab b* (sky=low b*, vegetation/terrain=high b*)
    """
    if channel == "b_star":
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2Lab)
        src = cv2.GaussianBlur(lab[:, :, 2], (5, 5), 0)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        src  = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = 0
    return boundary, mask


# ---------------------------------------------------------------------------
# Vectorised RANSAC with batched early stopping (unchanged from attempt 5)
# ---------------------------------------------------------------------------

def _best_hypothesis(
    points: np.ndarray,
    n_iter: int,
    inlier_threshold: float,
    rng: np.random.Generator,
    early_stop_ratio: float,
    batch_size: int,
) -> tuple[int, "np.ndarray | None"]:
    n_pts            = len(points)
    best_count       = 0
    best_mask        = None
    early_stop_count = int(n_pts * early_stop_ratio)

    done = 0
    while done < n_iter:
        b = min(batch_size, n_iter - done)

        idx_a = rng.integers(0, n_pts, size=b)
        idx_b = rng.integers(0, n_pts - 1, size=b)
        idx_b += (idx_b >= idx_a).astype(idx_b.dtype)

        pa = points[idx_a]
        pb = points[idx_b]

        delta  = (pb - pa).astype(np.float32)
        length = np.hypot(delta[:, 0], delta[:, 1])
        valid  = length > 1e-6
        safe   = np.where(valid, length, 1.0)
        vx = delta[:, 0] / safe
        vy = delta[:, 1] / safe

        diff_x = points[:, 0][None, :] - pa[:, 0:1]
        diff_y = points[:, 1][None, :] - pa[:, 1:2]
        dist   = np.abs(diff_x * vy[:, None] - diff_y * vx[:, None])

        inlier_masks  = dist < inlier_threshold
        inlier_counts = inlier_masks.sum(axis=1)
        inlier_counts[~valid] = 0

        top = int(inlier_counts.argmax())
        if inlier_counts[top] > best_count:
            best_count = int(inlier_counts[top])
            best_mask  = inlier_masks[top].copy()

        done += b
        if best_count >= early_stop_count:
            break

    return best_count, best_mask


# ---------------------------------------------------------------------------
# Row-scan fallback for near-zero-contrast frames
# ---------------------------------------------------------------------------

def _row_scan_horizon(image_bgr: np.ndarray) -> dict:
    """Return a flat (0°) horizon at the row with maximum horizontal gradient.

    Used when both classifier channels produce RANSAC confidence below
    _FALLBACK_CONF. A horizontal line at the right row is typically a better
    estimate than a wrong-angle RANSAC line on a featureless image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    row_energy = np.abs(gy).mean(axis=1)
    # Smooth over ~15 rows to suppress single-pixel noise
    row_energy = cv2.GaussianBlur(
        row_energy.reshape(-1, 1).astype(np.float32), (1, 15), 0
    ).flatten()
    peak_row = int(np.argmax(row_energy))
    return {
        "angle_deg":          0.0,
        "intercept_y_at_x0":  float(peak_row),
        "line":               (1.0, 0.0, 0.0, float(peak_row)),
        "confidence":         0.0,
        "inlier_count":       0,
        "mask":               None,
    }


# ---------------------------------------------------------------------------
# Per-channel pipeline helper
# ---------------------------------------------------------------------------

def _run_channel(
    image_bgr: np.ndarray,
    channel: str,
    ransac_iterations: int,
    inlier_threshold: float,
    early_stop_ratio: float,
    rng: np.random.Generator,
) -> "dict | None":
    boundary, mask = _extract_boundary(image_bgr, channel)
    ys, xs = np.where(boundary > 0)
    if len(xs) < 2:
        return None

    points = np.column_stack([xs, ys]).astype(np.float32)
    if len(points) > _MAX_BOUNDARY_PTS:
        idx    = rng.choice(len(points), _MAX_BOUNDARY_PTS, replace=False)
        points = points[idx]

    best_count, best_mask = _best_hypothesis(
        points, ransac_iterations, inlier_threshold, rng, early_stop_ratio, _BATCH_SIZE
    )
    if best_mask is None or best_count < 2:
        return None

    inlier_pts = points[best_mask].astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

    angle_deg = float(np.degrees(np.arctan2(vy, vx)))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg <= -90:
        angle_deg += 180

    intercept_y = float(y0 - (vy / vx) * x0) if abs(vx) > 1e-6 else float("nan")

    return {
        "angle_deg":          angle_deg,
        "intercept_y_at_x0":  intercept_y,
        "line":               (float(vx), float(vy), float(x0), float(y0)),
        "confidence":         best_count / len(points),
        "inlier_count":       best_count,
        "mask":               mask,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_horizon(
    image_bgr: np.ndarray,
    ransac_iterations: int    = _DEFAULT_ITER,
    inlier_threshold: float   = _INLIER_THRESHOLD,
    early_stop_ratio: float   = _EARLY_STOP_RATIO,
    random_seed: "int | None" = None,
) -> "dict | None":
    """Estimate the horizon via dual-channel classifier selection.

    Runs boundary extraction + RANSAC on both grayscale and Lab b*; returns
    whichever channel produced higher inlier confidence. Falls back to a
    row-scan estimate when both channels score below _FALLBACK_CONF.

    Returns a dict with keys: angle_deg, intercept_y_at_x0, line,
    confidence, inlier_count, mask. Returns None on complete failure.
    """
    global _rng
    rng = np.random.default_rng(random_seed) if random_seed is not None else _rng

    gray_result   = _run_channel(image_bgr, "gray",   ransac_iterations, inlier_threshold, early_stop_ratio, rng)
    b_star_result = _run_channel(image_bgr, "b_star", ransac_iterations, inlier_threshold, early_stop_ratio, rng)

    candidates = [r for r in (gray_result, b_star_result) if r is not None]
    if not candidates:
        return _row_scan_horizon(image_bgr)

    best = max(candidates, key=lambda r: r["confidence"])
    if best["confidence"] < _FALLBACK_CONF:
        return _row_scan_horizon(image_bgr)

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image",        type=Path)
    parser.add_argument("--out",        type=Path,  default=None)
    parser.add_argument("--iterations", type=int,   default=_DEFAULT_ITER)
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
        print(f"{args.image.name}: no horizon detected  ({elapsed_ms:.1f} ms)")
        return

    offset_str = "vertical" if np.isnan(result["intercept_y_at_x0"]) else f"{result['intercept_y_at_x0']:+.1f}px"
    print(
        f"{args.image.name}: conf={result['confidence']:.3f}"
        f"  angle={result['angle_deg']:+.2f}°  offset={offset_str}"
        f"  inliers={result['inlier_count']}  time={elapsed_ms:.1f} ms"
    )

    out_path = args.out or args.image.with_name(args.image.stem + "_horizon.jpg")
    out_img  = img.copy()
    h, w     = out_img.shape[:2]
    vx, vy, x0, y0 = result["line"]
    scale = max(h, w) * 2
    p1 = (int(round(x0 - scale * vx)), int(round(y0 - scale * vy)))
    p2 = (int(round(x0 + scale * vx)), int(round(y0 + scale * vy)))
    cv2.line(out_img, p1, p2, (0, 0, 255), 2)
    cv2.putText(
        out_img,
        f"angle={result['angle_deg']:+.2f}°  offset={offset_str}  conf={result['confidence']:.2f}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
    )
    cv2.imwrite(str(out_path), out_img)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
