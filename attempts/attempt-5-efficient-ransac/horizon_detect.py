"""
Attempt 5 — efficient RANSAC (N=1).

Changes from attempt 4:
  1. No clustering: for N=1 we argmax the per-hypothesis inlier counts and refit
     that single best inlier set. The O(n²) greedy cluster-and-refit pass is
     gone — it only helps when you need multiple well-separated candidates.
  2. Batched early stopping: hypotheses are scored in batches of BATCH_SIZE; once
     the running best exceeds EARLY_STOP_RATIO of boundary points the remaining
     batches are skipped. On easy frames this fires at 50–100 iterations instead
     of 300, cutting latency without touching accuracy.
  3. Module-level RNG: one Generator per process, reused across calls to avoid
     the construction overhead that attempt 4 paid on every frame.

Sky/ground classifier (grayscale + Otsu + morph close/open) and boundary
extraction are unchanged from attempts 2–4.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tuning knobs
# ---------------------------------------------------------------------------

_MAX_BOUNDARY_PTS = 400   # subsample cap before RANSAC
_DEFAULT_ITER     = 300   # maximum RANSAC iterations
_BATCH_SIZE       = 50    # iterations per batch (early-stop check cadence)
_EARLY_STOP_RATIO = 0.75  # stop if best hypothesis supports this fraction of pts
_INLIER_THRESHOLD = 3.0   # orthogonal-distance inlier threshold in pixels

# One Generator per process; recreated only when an explicit seed is passed.
_rng = np.random.default_rng()


# ---------------------------------------------------------------------------
# Boundary extraction (unchanged from attempts 2–4)
# ---------------------------------------------------------------------------

def _extract_boundary(image_bgr: np.ndarray):
    """Return (boundary_mask, sky_ground_mask)."""
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary[0, :] = boundary[-1, :] = boundary[:, 0] = boundary[:, -1] = 0
    return boundary, mask


# ---------------------------------------------------------------------------
# Vectorised RANSAC with batched early stopping
# ---------------------------------------------------------------------------

def _best_hypothesis(
    points: np.ndarray,
    n_iter: int,
    inlier_threshold: float,
    rng: np.random.Generator,
    early_stop_ratio: float,
    batch_size: int,
) -> tuple[int, "np.ndarray | None"]:
    """Return (best_inlier_count, best_inlier_mask) for the N=1 case.

    Hypotheses are scored batch-by-batch; the loop exits early once the
    running best covers early_stop_ratio of boundary points.
    """
    n_pts            = len(points)
    best_count       = 0
    best_mask        = None
    early_stop_count = int(n_pts * early_stop_ratio)

    done = 0
    while done < n_iter:
        b = min(batch_size, n_iter - done)

        # Sample b random pairs, avoiding i == j.
        idx_a = rng.integers(0, n_pts, size=b)
        idx_b = rng.integers(0, n_pts - 1, size=b)
        idx_b += (idx_b >= idx_a).astype(idx_b.dtype)

        pa = points[idx_a]   # (b, 2)
        pb = points[idx_b]   # (b, 2)

        # Direction unit vectors for all b hypotheses.
        delta  = (pb - pa).astype(np.float32)              # (b, 2)
        length = np.hypot(delta[:, 0], delta[:, 1])        # (b,)
        valid  = length > 1e-6
        safe   = np.where(valid, length, 1.0)
        vx = delta[:, 0] / safe   # (b,)
        vy = delta[:, 1] / safe   # (b,)

        # Orthogonal distance from every boundary point to every hypothesis.
        # diff_x[h, p] = points[p, 0] - pa[h, 0]  →  shape (b, n_pts)
        diff_x = points[:, 0][None, :] - pa[:, 0:1]
        diff_y = points[:, 1][None, :] - pa[:, 1:2]
        dist   = np.abs(diff_x * vy[:, None] - diff_y * vx[:, None])

        inlier_masks  = dist < inlier_threshold          # (b, n_pts) bool
        inlier_counts = inlier_masks.sum(axis=1)         # (b,)
        inlier_counts[~valid] = 0                        # reject degenerate pairs

        top = int(inlier_counts.argmax())
        if inlier_counts[top] > best_count:
            best_count = int(inlier_counts[top])
            best_mask  = inlier_masks[top].copy()        # own the array

        done += b
        if best_count >= early_stop_count:
            break

    return best_count, best_mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_horizon(
    image_bgr: np.ndarray,
    ransac_iterations: int   = _DEFAULT_ITER,
    inlier_threshold: float  = _INLIER_THRESHOLD,
    early_stop_ratio: float  = _EARLY_STOP_RATIO,
    random_seed: "int | None" = None,
) -> "dict | None":
    """Estimate the horizon line via efficient vectorised RANSAC.

    Returns a dict with:
      - angle_deg:          orientation in (-90, 90], degrees from the x-axis.
      - intercept_y_at_x0: y where the line crosses x=0 (NaN if near-vertical).
      - line:               (vx, vy, x0, y0) from cv2.fitLine.
      - confidence:         fraction of boundary pixels in the best inlier set.
      - inlier_count:       absolute number of supporting boundary pixels.
      - mask:               binary sky/ground mask.
    Returns None when no boundary can be extracted.
    """
    global _rng
    rng = np.random.default_rng(random_seed) if random_seed is not None else _rng

    boundary, mask = _extract_boundary(image_bgr)

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
    cv2.putText(out_img, f"angle={result['angle_deg']:+.2f}°  offset={offset_str}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(out_path), out_img)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
