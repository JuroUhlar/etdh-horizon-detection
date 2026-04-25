"""
tools/evaluate.py — run an attempt's detector over the Horizon-UAV dataset.

Usage:
    .venv/bin/python tools/evaluate.py attempts/attempt-1-otsu-column-scan
    .venv/bin/python tools/evaluate.py attempts/attempt-2-rotation-invariant --limit 50

Reports per-sample angular error, positional error (Hesse ρ), and sky-mask IoU,
plus aggregates (mean / P50 / P90 / max), a pass rate, and the worst offenders.

The evaluator is metric-definition-heavy on purpose: see docs/evaluation-metrics.md
for why we compare lines in Hesse normal form rather than via (slope, y-intercept).
"""

import argparse
import csv
import importlib.util
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "horizon_uav_dataset"

# Pass-rate tolerance — both must hold for a sample to "pass".
PASS_DTHETA_DEG = 5.0
PASS_DRHO_NORM = 0.05  # fraction of image height


# --------------------------- detector loading --------------------------- #

def load_detector(attempt_dir: Path):
    """Import horizon_detect.py from an attempt directory as a module.

    Each attempt is a standalone script with a detect_horizon(image_bgr)
    function; we don't want to force attempts to become installable packages
    just so we can call them from here, so we load them dynamically.
    """
    script = attempt_dir / "horizon_detect.py"
    if not script.exists():
        raise SystemExit(f"No horizon_detect.py in {attempt_dir}")
    spec = importlib.util.spec_from_file_location(f"attempt_{attempt_dir.name}", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "detect_horizon"):
        raise SystemExit(f"{script} does not expose detect_horizon()")
    return module.detect_horizon


def normalise_output(raw):
    """Convert any attempt's return shape into ((vx, vy, x0, y0), mask_or_None).

    Attempts can return:
      - None  — no horizon detected.
      - (slope_deg, intercept_px, mask)  — attempt 1 style.
      - dict with 'line' = (vx, vy, x0, y0) and 'mask'  — attempt 2 style.
    """
    if raw is None:
        return None, None
    if isinstance(raw, list):
        if not raw:
            return None, None
        return raw[0]["line"], raw[0].get("mask")
    if isinstance(raw, dict):
        return raw["line"], raw.get("mask")
    if isinstance(raw, tuple) and len(raw) == 3:
        slope_deg, intercept_px, mask = raw
        theta = math.radians(slope_deg)
        return (math.cos(theta), math.sin(theta), 0.0, intercept_px), mask
    raise ValueError(f"Unsupported detector return type: {type(raw).__name__}")


# --------------------------- line geometry --------------------------- #

def line_from_slope_offset(slope: float, offset_normalised: float, image_height: int):
    """GT label → line in (vx, vy, x0, y0) form.

    The label says `y = slope*x + c` where `c = offset_normalised * image_height`.
    We use direction (1, slope) and a point (0, c) on the line; cv2.fitLine-style.
    """
    c_px = offset_normalised * image_height
    return (1.0, slope, 0.0, c_px)


def hesse_canonical(vx: float, vy: float, x0: float, y0: float):
    """Canonical Hesse form (nx, ny, rho) with ny >= 0.

    Every line has two direction-vector representations (direction and its
    negative) that encode the same line but produce opposite normals and
    opposite signs of rho. We canonicalise to ny >= 0 so that two
    representations of the same line produce the same (nx, ny, rho).
    """
    L = math.hypot(vx, vy)
    nx, ny = -vy / L, vx / L       # normal = direction rotated +90°
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny          # flip into upper half-plane
    rho = nx * x0 + ny * y0
    return nx, ny, rho


def angular_error_deg(line_a, line_b) -> float:
    """Angular distance between two lines, in [0, 90] degrees.

    Lines are direction-agnostic (a line pointing one way is the same as
    the same line pointing the other way), so we fold to mod 180° and then
    take the shorter arc.
    """
    theta_a = math.degrees(math.atan2(line_a[1], line_a[0]))
    theta_b = math.degrees(math.atan2(line_b[1], line_b[0]))
    raw = abs(theta_a - theta_b) % 180.0
    return min(raw, 180.0 - raw)


def positional_error_px(line_pred, line_gt) -> float:
    """|Δρ| between predicted and GT line in canonical Hesse form.

    Intuition: if two lines are parallel, |Δρ| is literally the distance
    between them in pixels. For non-parallel lines it's a signed difference
    in where each line sits relative to the origin along the (canonicalised)
    normal direction — small when the lines are close, large otherwise.
    """
    _, _, rho_pred = hesse_canonical(*line_pred)
    _, _, rho_gt = hesse_canonical(*line_gt)
    return abs(rho_pred - rho_gt)


def iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks."""
    inter = np.logical_and(pred, gt).sum(dtype=np.int64)
    union = np.logical_or(pred, gt).sum(dtype=np.int64)
    return float(inter) / float(union) if union else 1.0


# --------------------------- evaluation loop --------------------------- #

@dataclass
class SampleResult:
    filename: str
    delta_theta_deg: Optional[float]
    delta_rho_px: Optional[float]
    delta_rho_norm: Optional[float]
    iou: Optional[float]
    latency_ms: float
    failed: bool


def evaluate(attempt_dir: Path, dataset_dir: Path, limit: Optional[int] = None):
    detect = load_detector(attempt_dir)

    with (dataset_dir / "label.csv").open() as f:
        labels = list(csv.DictReader(f))
    if limit is not None:
        labels = labels[:limit]

    total = len(labels)
    results: list[SampleResult] = []
    for i, row in enumerate(labels, 1):
        print(f"\r  {i}/{total}", end="", flush=True)
        filename = row["filename"]
        slope = float(row["slope"])
        offset = float(row["offset"])

        img_path = dataset_dir / "images" / filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: could not read {img_path}", file=sys.stderr)
            continue
        H, W = img.shape[:2]

        t0 = time.perf_counter()
        raw = detect(img)
        latency_ms = (time.perf_counter() - t0) * 1000

        line_pred, mask_pred = normalise_output(raw)
        if line_pred is None:
            results.append(SampleResult(filename, None, None, None, None, latency_ms, failed=True))
            continue

        line_gt = line_from_slope_offset(slope, offset, H)
        d_theta = angular_error_deg(line_pred, line_gt)
        d_rho = positional_error_px(line_pred, line_gt)
        d_rho_norm = d_rho / H

        iou_val: Optional[float] = None
        if mask_pred is not None:
            mask_path = dataset_dir / "masks" / "sky" / (Path(filename).stem + ".png")
            mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_gt is not None:
                gt_bool = mask_gt > 127
                pred_bool = mask_pred > 127 if mask_pred.dtype != bool else mask_pred
                iou_val = iou_binary(pred_bool, gt_bool)

        results.append(SampleResult(filename, d_theta, d_rho, d_rho_norm, iou_val, latency_ms, failed=False))

    print()  # newline after progress
    return results


# --------------------------- reporting --------------------------- #

def _pct(vals, p):
    return float(np.percentile(vals, p)) if len(vals) else float("nan")


def _row(label, values, fmt="{:8.3f}"):
    arr = np.asarray(values)
    mean = arr.mean()
    p50 = np.percentile(arr, 50)
    p90 = np.percentile(arr, 90)
    vmax = arr.max()
    print(
        f"  {label:<32s}"
        f"  mean={fmt.format(mean)}"
        f"  P50={fmt.format(p50)}"
        f"  P90={fmt.format(p90)}"
        f"  max={fmt.format(vmax)}"
    )


def print_report(results: list[SampleResult], attempt_name: str):
    ok = [r for r in results if not r.failed]
    failed = [r for r in results if r.failed]
    n = len(results)
    print(f"\n=== {attempt_name}  —  {len(ok)} ok / {len(failed)} failed / {n} total ===\n")
    if not ok:
        print("  all samples failed; nothing to aggregate.")
        return

    _row("Δθ (deg)",                 [r.delta_theta_deg for r in ok])
    _row("Δρ (px, Hesse)",           [r.delta_rho_px for r in ok])
    _row("Δρ / image_height",        [r.delta_rho_norm for r in ok])

    iou_vals = [r.iou for r in ok if r.iou is not None]
    if iou_vals:
        _row("sky-mask IoU",         iou_vals)

    _row("latency (ms)",             [r.latency_ms for r in ok])

    passed = sum(
        1 for r in ok
        if r.delta_theta_deg < PASS_DTHETA_DEG and r.delta_rho_norm < PASS_DRHO_NORM
    )
    print(
        f"\n  pass rate  (Δθ<{PASS_DTHETA_DEG:.0f}°  &  Δρ/H<{PASS_DRHO_NORM*100:.0f}%)  "
        f"=  {passed}/{len(ok)}  ({passed/len(ok)*100:.1f}%)"
    )

    print("\n  worst 5 by Δθ:")
    for r in sorted(ok, key=lambda r: r.delta_theta_deg, reverse=True)[:5]:
        iou_str = f"  IoU={r.iou:.3f}" if r.iou is not None else ""
        print(
            f"    {r.filename[:58]:58s}  "
            f"Δθ={r.delta_theta_deg:6.2f}°  "
            f"Δρ={r.delta_rho_px:7.2f}px"
            f"{iou_str}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("attempt", type=Path, help="Path to an attempt folder, e.g. attempts/attempt-2-rotation-invariant")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to Horizon-UAV dataset root")
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples (for quick iteration)")
    args = p.parse_args()

    t_start = time.perf_counter()
    results = evaluate(args.attempt, args.dataset, limit=args.limit)
    elapsed = time.perf_counter() - t_start

    print_report(results, attempt_name=args.attempt.name)
    print(f"\n  total wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
