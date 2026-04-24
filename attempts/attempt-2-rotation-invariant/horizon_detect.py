"""
Attempt 2 — rotation-invariant boundary extraction + robust line fit.

Upgrades attempt 1 in two ways:
  1. Replace the column-scan (which assumed "sky is on top") with a boundary
     extraction via morphological gradient, which is rotation-invariant.
  2. Replace np.polyfit (least-squares on y, fails for near-vertical lines and
     is outlier-sensitive) with cv2.fitLine + Huber loss. fitLine minimises
     *orthogonal* distance, so it works at any orientation; Huber caps the
     influence of outliers.

Kept from attempt 1: grayscale + blur + Otsu + morphological closing. Otsu is
already orientation-invariant (it only looks at the histogram), so the upstream
half of the pipeline did not need changing.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def detect_horizon(image_bgr: np.ndarray):
    """
    Estimate the horizon as a line at any angle (including near-vertical).

    Returns a dict with:
      - angle_deg:  orientation in (-90, 90], degrees from the x-axis.
      - intercept_y_at_x0: y where the line crosses x=0, or NaN for
                           near-vertical lines (no meaningful y-intercept).
      - line: (vx, vy, x0, y0) from cv2.fitLine — usable for drawing at
              any orientation, even when the y-intercept form breaks down.
      - mask: the binary sky/ground mask.
    Returns None if no boundary can be extracted.
    """
    # --- 1. Sky/ground mask (carried over from attempt 1) ---
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _thresh, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    # Close fills holes inside the sky region; open removes small isolated specks.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # --- 2. Extract the sky/ground boundary as a point cloud ---
    # Morphological gradient = dilation - erosion. Produces a 1–2 pixel ridge
    # along the region boundary regardless of orientation.
    gradient_kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, gradient_kernel)

    # Where the sky mask touches the image edge, the gradient reports a
    # "boundary" along the frame itself — which is not the horizon. Zero out
    # the 1-pixel-wide image border so those don't bias the fit.
    boundary[0, :] = 0
    boundary[-1, :] = 0
    boundary[:, 0] = 0
    boundary[:, -1] = 0

    ys, xs = np.where(boundary > 0)
    if len(xs) < 2:
        return None

    # --- 3. Robust orthogonal line fit ---
    # cv2.fitLine minimises orthogonal distance, so it handles any orientation.
    # DIST_HUBER behaves like L2 near zero and like L1 for large residuals,
    # blunting the effect of misclassified-patch outliers.
    points = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()

    # Direction vector -> angle in (-90, 90]. Horizons are symmetric under a
    # 180° flip of the direction, e.g. +170° and -10° describe the same line.
    angle_deg = float(np.degrees(np.arctan2(vy, vx)))
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg <= -90:
        angle_deg += 180

    # y = m*x + b form, when the line isn't near-vertical.
    if abs(vx) > 1e-6:
        slope = vy / vx
        intercept_y = float(y0 - slope * x0)
    else:
        intercept_y = float("nan")

    return {
        "angle_deg": angle_deg,
        "intercept_y_at_x0": intercept_y,
        "line": (float(vx), float(vy), float(x0), float(y0)),
        "mask": mask,
    }


def draw_horizon(image_bgr: np.ndarray, line, angle_deg: float, intercept_y: float) -> np.ndarray:
    """Draw the horizon line across the image using the direction-vector form."""
    out = image_bgr.copy()
    h, w = out.shape[:2]
    vx, vy, x0, y0 = line
    # Extend far in both directions along the line; cv2.line clips to frame.
    scale = max(h, w) * 2
    p1 = (int(round(x0 - scale * vx)), int(round(y0 - scale * vy)))
    p2 = (int(round(x0 + scale * vx)), int(round(y0 + scale * vy)))
    cv2.line(out, p1, p2, (0, 0, 255), 2)

    if np.isnan(intercept_y):
        label = f"angle={angle_deg:+.2f} deg  offset=(vertical)"
    else:
        label = f"angle={angle_deg:+.2f} deg  offset={intercept_y:+.1f}px"
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument("--out", type=Path, default=None, help="Output path for annotated image")
    args = parser.parse_args()

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    t0 = time.perf_counter()
    result = detect_horizon(img)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if result is None:
        print(f"{args.image.name}: no horizon detected  (time={elapsed_ms:.1f}ms)")
        return

    angle_deg = result["angle_deg"]
    intercept_y = result["intercept_y_at_x0"]
    line = result["line"]

    offset_str = "vertical line" if np.isnan(intercept_y) else f"{intercept_y:+.1f}px"
    print(
        f"{args.image.name}: angle={angle_deg:+.2f} deg  offset={offset_str}  "
        f"time={elapsed_ms:.1f}ms"
    )

    out_path = args.out or args.image.with_name(args.image.stem + "_horizon.jpg")
    annotated = draw_horizon(img, line, angle_deg, intercept_y)
    cv2.imwrite(str(out_path), annotated)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
