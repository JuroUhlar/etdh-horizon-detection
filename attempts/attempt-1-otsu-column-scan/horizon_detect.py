"""
Classical horizon detection baseline (Otsu + morphology + line fit).

Reads an image, estimates the horizon line as y = m*x + b, and writes an
annotated copy with the line drawn across it. Prints the angle (in degrees)
and the vertical intercept (in pixels) to stdout.

This is the simplest pipeline from the research doc — fast, no ML, and a
useful baseline to beat before reaching for fancier methods.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def detect_horizon(image_bgr: np.ndarray):
    """
    Estimate the horizon line in a BGR image.

    Returns (slope_deg, intercept_px, sky_mask) or None if no horizon is found.
    - slope_deg: angle of the line relative to the x-axis, in degrees.
                 Positive means the right side of the horizon is lower than the left.
    - intercept_px: y value of the line at x=0, in pixels (0 = top of image).
    - sky_mask: uint8 array, 255 where the algorithm thinks the pixel is sky.
    """
    # Step 1 — grayscale. Sky/ground differ mostly in brightness, so one channel is enough
    # and ~3x cheaper than working on BGR.
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Step 2 — blur. A 5x5 Gaussian kills high-frequency texture (leaves, waves, gravel)
    # that would otherwise confuse the thresholding step. Larger kernels smear the horizon
    # itself; 5x5 is a good starting point for ~480p input.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3 — Otsu. Instead of guessing a threshold, Otsu picks the one that best separates
    # the pixel histogram into two groups. Works great when the histogram is bimodal
    # (clear sky vs. clear ground); struggles when they overlap (overcast sky, snow, etc.).
    # After this, sky_mask is 255 for bright pixels (sky) and 0 for dark (ground).
    _thresh_value, sky_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 4 — morphological closing. Dilate then erode with a 9x9 kernel. This fills small
    # dark specks inside the sky region (e.g. a dark cloud) without shrinking the sky's
    # outer boundary. It does NOT help with large misclassifications — that needs a smarter
    # algorithm.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel)

    # Step 5 — find the horizon row per column.
    # Trick: np.argmax on a boolean array returns the index of the FIRST True.
    # We build `ground_mask` so True = "this pixel is ground". For each column, the first
    # True going top-down is the first ground pixel — i.e. the horizon row in that column.
    ground_mask = sky_mask == 0
    horizon_y_per_column = np.argmax(ground_mask, axis=0)

    # Edge case: columns that are all-sky or all-ground don't contain a transition, and
    # argmax returns 0 for them — which would pull the fit toward the top of the image.
    # Keep only columns that contain BOTH sky and ground.
    has_sky = (~ground_mask).any(axis=0)
    has_ground = ground_mask.any(axis=0)
    valid = has_sky & has_ground

    if valid.sum() < 2:
        return None

    xs = np.where(valid)[0]
    ys = horizon_y_per_column[valid]

    # Step 6 — least-squares line fit. np.polyfit(..., deg=1) returns [slope, intercept].
    # Downside: a single outlier (e.g. a tall tree poking into the sky) tugs the line
    # toward it. RANSAC or Theil–Sen would be more robust — good next upgrade.
    slope, intercept = np.polyfit(xs, ys, 1)

    slope_deg = float(np.degrees(np.arctan(slope)))
    intercept_px = float(intercept)
    return slope_deg, intercept_px, sky_mask


def draw_horizon(image_bgr: np.ndarray, slope_deg: float, intercept_px: float) -> np.ndarray:
    """Return a copy of the image with the horizon line and readout drawn on it."""
    out = image_bgr.copy()
    h, w = out.shape[:2]

    # Convert back from (angle, intercept) to two endpoints at x=0 and x=w-1.
    slope = np.tan(np.radians(slope_deg))
    x0, x1 = 0, w - 1
    y0 = int(round(slope * x0 + intercept_px))
    y1 = int(round(slope * x1 + intercept_px))

    cv2.line(out, (x0, y0), (x1, y1), (0, 0, 255), 2)  # red line
    label = f"angle={slope_deg:+.2f} deg  offset={intercept_px:.1f}px"
    cv2.putText(out, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to input image")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for the annotated image (default: <input>_horizon.jpg)",
    )
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

    slope_deg, intercept_px, _mask = result
    print(
        f"{args.image.name}: angle={slope_deg:+.2f} deg  "
        f"offset={intercept_px:.1f}px  time={elapsed_ms:.1f}ms"
    )

    out_path = args.out or args.image.with_name(args.image.stem + "_horizon.jpg")
    annotated = draw_horizon(img, slope_deg, intercept_px)
    cv2.imwrite(str(out_path), annotated)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
