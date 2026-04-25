"""
tools/evaluate.py — run an attempt's detector over a horizon dataset.

Usage:
    .venv/bin/python tools/evaluate.py attempts/attempt-1-otsu-column-scan
    .venv/bin/python tools/evaluate.py attempts/attempt-2-rotation-invariant --limit 50
    .venv/bin/python tools/evaluate.py attempts/attempt-3-top-n-ransac --dataset data/video_clips_ukraine_atv

Reports per-sample angular error, positional error (Hesse ρ), and sky-mask IoU,
plus aggregates (mean / P50 / P90 / max), a pass rate, and the worst offenders.

Datasets may include a has_horizon column (data/video_clips_ukraine_atv) or omit
it (data/horizon_uav_dataset, where every frame has a horizon). When present, the
report adds a confusion matrix and pass-rate folds in the no-horizon agreement.

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
    """Convert any attempt's return shape into (line, mask, no_horizon).

    Returns:
      line: (vx, vy, x0, y0) tuple, or None when there is no horizon line to compare.
      mask: optional sky mask (numpy array), or None.
      no_horizon: True if the detector deliberately reported "no horizon present".
                  False if the detector returned a line OR returned None (failure).
                  This distinguishes "I correctly abstained" from "I crashed/gave up".

    Attempts can return:
      - None                                — failure / detector gave up.
      - "no_horizon" (str)                  — deliberate abstention (sky-only / ground-only).
      - {"no_horizon": True, "mask": ...}   — same, in dict form, with optional mask.
      - (slope_deg, intercept_px, mask)     — attempt 1 style.
      - {"line": (vx, vy, x0, y0), ...}     — attempt 2 / 3 style.
    """
    if raw is None:
        return None, None, False
    if isinstance(raw, str) and raw == "no_horizon":
        return None, None, True
    if isinstance(raw, dict):
        if raw.get("no_horizon"):
            return None, raw.get("mask"), True
        return raw["line"], raw.get("mask"), False
    if isinstance(raw, tuple) and len(raw) == 3:
        slope_deg, intercept_px, mask = raw
        theta = math.radians(slope_deg)
        return (math.cos(theta), math.sin(theta), 0.0, intercept_px), mask, False
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
    gt_has_horizon: bool
    pred_has_horizon: bool        # False ONLY when detector deliberately said "no_horizon"
    delta_theta_deg: Optional[float]
    delta_rho_px: Optional[float]
    delta_rho_norm: Optional[float]
    iou: Optional[float]
    latency_ms: float
    failed: bool                  # True when detector returned None (crashed / gave up)


def _load_labels(csv_path: Path) -> list[dict]:
    """Read label.csv tolerating both 3-col (legacy) and 4-col (with has_horizon) schemas."""
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        has_col = "has_horizon" in (reader.fieldnames or [])
        rows = []
        for row in reader:
            hh = (row["has_horizon"].strip().lower() == "true") if has_col else True
            rows.append({
                "filename": row["filename"],
                "has_horizon": hh,
                "slope": float(row["slope"]) if hh and row.get("slope") else None,
                "offset": float(row["offset"]) if hh and row.get("offset") else None,
            })
    return rows


def evaluate(attempt_dir: Path, dataset_dir: Path, limit: Optional[int] = None):
    detect = load_detector(attempt_dir)

    labels = _load_labels(dataset_dir / "label.csv")
    if limit is not None:
        labels = labels[:limit]

    results: list[SampleResult] = []
    for row in labels:
        filename = row["filename"]
        gt_hh = row["has_horizon"]

        img_path = dataset_dir / "images" / filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: could not read {img_path}", file=sys.stderr)
            continue
        H, W = img.shape[:2]

        t0 = time.perf_counter()
        raw = detect(img)
        latency_ms = (time.perf_counter() - t0) * 1000

        line_pred, mask_pred, pred_no_horizon = normalise_output(raw)

        # Distinguish detector failure (returned None, no decision) from deliberate
        # no-horizon. Only the former counts as "failed" in reporting.
        if line_pred is None and not pred_no_horizon:
            results.append(SampleResult(
                filename, gt_hh, True, None, None, None, None, latency_ms, failed=True,
            ))
            continue

        pred_hh = not pred_no_horizon

        # Line errors are only defined when both label and prediction are horizons.
        d_theta = d_rho = d_rho_norm = None
        if gt_hh and pred_hh:
            line_gt = line_from_slope_offset(row["slope"], row["offset"], H)
            d_theta = angular_error_deg(line_pred, line_gt)
            d_rho = positional_error_px(line_pred, line_gt)
            d_rho_norm = d_rho / H

        # IoU is comparable whenever we have both masks, regardless of has_horizon.
        iou_val: Optional[float] = None
        if mask_pred is not None:
            mask_path = dataset_dir / "masks" / "sky" / (Path(filename).stem + ".png")
            mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_gt is not None:
                gt_bool = mask_gt > 127
                pred_bool = mask_pred > 127 if mask_pred.dtype != bool else mask_pred
                iou_val = iou_binary(pred_bool, gt_bool)

        results.append(SampleResult(
            filename, gt_hh, pred_hh, d_theta, d_rho, d_rho_norm, iou_val, latency_ms, failed=False,
        ))

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

    # Confusion matrix over has_horizon. Trigger off ground truth in ALL results (including
    # failed rows): if every no-horizon sample crashed the detector, those rows live in `failed`
    # not `ok`, and we'd otherwise hide the matrix exactly when it would be most informative.
    has_no_horizon_labels = any(not r.gt_has_horizon for r in results)
    tp = sum(1 for r in ok if r.gt_has_horizon and r.pred_has_horizon)
    fn = sum(1 for r in ok if r.gt_has_horizon and not r.pred_has_horizon)
    fp = sum(1 for r in ok if not r.gt_has_horizon and r.pred_has_horizon)
    tn = sum(1 for r in ok if not r.gt_has_horizon and not r.pred_has_horizon)
    if has_no_horizon_labels:
        # Break failures down by GT class so silent-on-no-horizon detectors aren't hidden.
        failed_gt_horizon = sum(1 for r in failed if r.gt_has_horizon)
        failed_gt_no_horizon = sum(1 for r in failed if not r.gt_has_horizon)
        print("  has_horizon confusion matrix:")
        print(f"    TP (gt=horizon, pred=horizon)         = {tp}")
        print(f"    FN (gt=horizon, pred=no_horizon)      = {fn}")
        print(f"    FP (gt=no_horizon, pred=horizon)      = {fp}")
        print(f"    TN (gt=no_horizon, pred=no_horizon)   = {tn}")
        if failed:
            print(f"    failed (no decision)                  = {len(failed)} "
                  f"(gt=horizon: {failed_gt_horizon}, gt=no_horizon: {failed_gt_no_horizon})")
        print()

    # Line errors are only defined for true positives (both sides agree there's a horizon).
    line_rows = [r for r in ok if r.delta_theta_deg is not None]
    if line_rows:
        _row("Δθ (deg)",                 [r.delta_theta_deg for r in line_rows])
        _row("Δρ (px, Hesse)",           [r.delta_rho_px for r in line_rows])
        _row("Δρ / image_height",        [r.delta_rho_norm for r in line_rows])

    iou_vals = [r.iou for r in ok if r.iou is not None]
    if iou_vals:
        _row("sky-mask IoU",         iou_vals)

    _row("latency (ms)",             [r.latency_ms for r in ok])

    # Pass = correct has_horizon classification AND (when both horizon) line errors within threshold.
    def passes(r: SampleResult) -> bool:
        if r.gt_has_horizon != r.pred_has_horizon:
            return False
        if not r.gt_has_horizon:
            return True  # both correctly say no-horizon; no line to score
        return r.delta_theta_deg < PASS_DTHETA_DEG and r.delta_rho_norm < PASS_DRHO_NORM

    passed = sum(1 for r in ok if passes(r))
    pass_total = len(results)  # failed samples count against the pass rate
    pass_label = (
        f"  pass rate  (correct has_horizon  &  Δθ<{PASS_DTHETA_DEG:.0f}°  &  Δρ/H<{PASS_DRHO_NORM*100:.0f}%)"
        if has_no_horizon_labels
        else f"  pass rate  (Δθ<{PASS_DTHETA_DEG:.0f}°  &  Δρ/H<{PASS_DRHO_NORM*100:.0f}%)"
    )
    print(
        f"\n{pass_label}"
        f"  =  {passed}/{pass_total}  ({passed/pass_total*100:.1f}%)"
    )

    if line_rows:
        print("\n  worst 5 by Δθ:")
        for r in sorted(line_rows, key=lambda r: r.delta_theta_deg, reverse=True)[:5]:
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

    results = evaluate(args.attempt, args.dataset, limit=args.limit)
    print_report(results, attempt_name=args.attempt.name)


if __name__ == "__main__":
    main()
