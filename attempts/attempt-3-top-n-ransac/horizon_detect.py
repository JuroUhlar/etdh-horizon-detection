"""
Attempt 3 — top-N horizon candidates via RANSAC + Huber refit.

Upgrades attempt 2 in two ways:
  1. Replace the single cv2.fitLine call with RANSAC: repeatedly sample 2
     boundary points, count orthogonal-distance inliers, cluster overlapping
     hypotheses, and refit each cluster with cv2.fitLine + Huber. This surfaces
     the N most supported horizon lines rather than just one.
  2. Each result carries a confidence score (fraction of boundary pixels that
     support that line), giving a natural ranking.

Kept from attempt 2: grayscale + blur + Otsu + morphological closing +
morphological-gradient boundary extraction. Those are all orientation-invariant
and did not need changing.

A random_seed parameter is exposed so the RANSAC sampling can be made
reproducible when needed.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_boundary(image_bgr: np.ndarray):
    """Return (boundary_mask, sky_ground_mask)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _thresh, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    gradient_kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, gradient_kernel)

    # Zero out the frame border — those pixels are not the horizon.
    boundary[0, :] = 0
    boundary[-1, :] = 0
    boundary[:, 0] = 0
    boundary[:, -1] = 0

    return boundary, mask


def _ransac_hypotheses(
    points: np.ndarray,
    n_iter: int,
    inlier_threshold: float,
    rng: np.random.Generator,
) -> list:
    """
    Draw n_iter random 2-point lines, score each by inlier count.

    Returns list of (inlier_count: int, inlier_mask: ndarray[bool]).
    """
    n = len(points)
    hypotheses = []

    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        delta = points[j] - points[i]
        length = float(np.hypot(delta[0], delta[1]))
        if length < 1e-6:
            continue

        vx, vy = delta / length
        diff = points - points[i]
        # Orthogonal distance: |diff × direction|
        dist = np.abs(diff[:, 0] * vy - diff[:, 1] * vx)

        inlier_mask = dist < inlier_threshold
        inlier_count = int(inlier_mask.sum())
        if inlier_count >= 2:
            hypotheses.append((inlier_count, inlier_mask))

    return hypotheses


def _cluster_and_refit(
    hypotheses: list,
    points: np.ndarray,
    overlap_threshold: float = 0.7,
) -> list:
    """
    Greedy clustering by inlier-set overlap, then refit each cluster with
    cv2.fitLine + Huber.  Returns result dicts sorted by confidence desc.
    """
    hypotheses.sort(key=lambda x: x[0], reverse=True)
    total = len(points)
    used = [False] * len(hypotheses)
    results = []

    for i, (count_i, mask_i) in enumerate(hypotheses):
        if used[i]:
            continue
        used[i] = True

        consensus = mask_i.copy()
        for j in range(i + 1, len(hypotheses)):
            if used[j]:
                continue
            count_j, mask_j = hypotheses[j]
            overlap = int((mask_i & mask_j).sum()) / max(count_i, count_j)
            if overlap > overlap_threshold:
                consensus |= mask_j
                used[j] = True

        inlier_pts = points[consensus].astype(np.float32)
        if len(inlier_pts) < 2:
            continue

        vx, vy, x0, y0 = cv2.fitLine(inlier_pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

        angle_deg = float(np.degrees(np.arctan2(vy, vx)))
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg <= -90:
            angle_deg += 180

        intercept_y = float(y0 - (vy / vx) * x0) if abs(vx) > 1e-6 else float("nan")
        inlier_count = int(consensus.sum())

        results.append({
            "angle_deg": angle_deg,
            "intercept_y_at_x0": intercept_y,
            "line": (float(vx), float(vy), float(x0), float(y0)),
            "confidence": inlier_count / total,
            "inlier_count": inlier_count,
        })

    results.sort(key=lambda r: r["confidence"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_horizon(
    image_bgr: np.ndarray,
    n: int = 1,
    ransac_iterations: int = 500,
    inlier_threshold: float = 3.0,
    random_seed: int | None = None,
) -> list[dict]:
    """
    Estimate the top-n most likely horizon lines, sorted by confidence desc.

    Each result dict contains:
      - angle_deg:          orientation in (-90, 90], degrees from the x-axis.
      - intercept_y_at_x0: y where the line crosses x=0 (NaN if near-vertical).
      - line:               (vx, vy, x0, y0) from cv2.fitLine — usable for
                            drawing at any orientation.
      - confidence:         fraction of boundary pixels supporting this line (0–1).
      - inlier_count:       absolute number of supporting boundary pixels.
      - mask:               binary sky/ground mask (shared across all results).

    Returns an empty list if no boundary can be extracted.

    Args:
        n:                  number of candidates to return.
        ransac_iterations:  how many random 2-point samples to draw.
        inlier_threshold:   max orthogonal distance (px) to count as an inlier.
        random_seed:        seed for RANSAC RNG; None = non-deterministic.
    """
    boundary, mask = _extract_boundary(image_bgr)

    ys, xs = np.where(boundary > 0)
    if len(xs) < 2:
        return []

    points = np.column_stack([xs, ys]).astype(np.float32)
    rng = np.random.default_rng(random_seed)

    hypotheses = _ransac_hypotheses(points, ransac_iterations, inlier_threshold, rng)
    if not hypotheses:
        return []

    results = _cluster_and_refit(hypotheses, points)
    for r in results:
        r["mask"] = mask

    return results[:n]


def draw_horizon(image_bgr: np.ndarray, results: list[dict]) -> np.ndarray:
    """Draw all horizon candidates. Best (highest confidence) is drawn brightest."""
    out = image_bgr.copy()
    h, w = out.shape[:2]

    # Best → red, then orange, yellow, green, cyan
    palette = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 128), (0, 255, 0)]

    for rank, r in enumerate(results):
        vx, vy, x0, y0 = r["line"]
        color = palette[rank % len(palette)]
        thickness = max(1, 3 - rank)

        scale = max(h, w) * 2
        p1 = (int(round(x0 - scale * vx)), int(round(y0 - scale * vy)))
        p2 = (int(round(x0 + scale * vx)), int(round(y0 + scale * vy)))
        cv2.line(out, p1, p2, color, thickness)

        offset = r["intercept_y_at_x0"]
        offset_str = "vertical" if np.isnan(offset) else f"{offset:+.1f}px"
        label = (
            f"#{rank + 1}  conf={r['confidence']:.2f}"
            f"  angle={r['angle_deg']:+.2f}deg  offset={offset_str}"
        )
        cv2.putText(out, label, (10, 25 + rank * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--out", type=Path, default=None, help="Output path for annotated image")
    parser.add_argument("--n", type=int, default=1, help="Number of horizon candidates (default: 1)")
    parser.add_argument("--iterations", type=int, default=500, help="RANSAC iterations (default: 500)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Inlier threshold in pixels (default: 3.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    t0 = time.perf_counter()
    results = detect_horizon(
        img,
        n=args.n,
        ransac_iterations=args.iterations,
        inlier_threshold=args.threshold,
        random_seed=args.seed,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if not results:
        print(f"{args.image.name}: no horizon detected  (time={elapsed_ms:.1f}ms)")
        return

    for rank, r in enumerate(results):
        offset = r["intercept_y_at_x0"]
        offset_str = "vertical" if np.isnan(offset) else f"{offset:+.1f}px"
        print(
            f"{args.image.name} #{rank + 1}: "
            f"conf={r['confidence']:.3f}  angle={r['angle_deg']:+.2f}deg  "
            f"offset={offset_str}  inliers={r['inlier_count']}  "
            f"time={elapsed_ms:.1f}ms"
        )

    out_path = args.out or args.image.with_name(args.image.stem + "_horizon.jpg")
    annotated = draw_horizon(img, results)
    cv2.imwrite(str(out_path), annotated)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
