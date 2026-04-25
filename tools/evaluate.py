"""
tools/evaluate.py — run an attempt's detector over a horizon dataset.

Usage:
    .venv/bin/python tools/evaluate.py attempts/attempt-1-otsu-column-scan
    .venv/bin/python tools/evaluate.py attempts/attempt-2-rotation-invariant --limit 50
    .venv/bin/python tools/evaluate.py attempts/attempt-3-top-n-ransac --dataset data/video_clips_ukraine_atv
    .venv/bin/python tools/evaluate.py attempts/attempt-3-top-n-ransac --seed 0

Reports per-sample angular error, positional error (Hesse rho), and sky-mask IoU,
plus aggregates, a pass rate, and the worst offenders.

Datasets may include a has_horizon column (data/video_clips_ukraine_atv) or omit
it (data/horizon_uav_dataset, where every frame has a horizon). When present, the
report adds a confusion matrix and pass-rate folds in the no-horizon agreement.

The evaluator is metric-definition-heavy on purpose: see docs/evaluation-metrics.md
for why we compare lines in Hesse normal form rather than via (slope, y-intercept).
"""

import argparse
import csv
import importlib.util
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ANSI colours — disabled automatically when stdout is not a TTY
# (e.g. redirected to a file) or when NO_COLOR env var is set.
# ---------------------------------------------------------------------------
def _supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class C:
    """Terminal colour codes."""

    if _supports_color():
        PASS = "\033[92m"
        FAIL = "\033[91m"
        WARN = "\033[93m"
        DIM = "\033[2m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
    else:
        PASS = FAIL = WARN = DIM = BOLD = RESET = ""


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "horizon_uav_dataset"

# Pass-rate tolerance — both must hold for a sample to "pass".
PASS_DTHETA_DEG = 5.0
PASS_DRHO_NORM = 0.05  # fraction of image height
LATENCY_BUDGET_MS = 1000 / 15  # 66.7 ms = 15 FPS


# --------------------------- detector loading --------------------------- #

def load_detector(attempt_dir: Path):
    """Import horizon_detect.py from an attempt directory as a module."""
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
    """Convert any attempt's return shape into (line, mask, no_horizon)."""
    if raw is None:
        return None, None, False
    if isinstance(raw, str) and raw == "no_horizon":
        return None, None, True
    if isinstance(raw, list):
        if not raw:
            return None, None, False
        return normalise_output(raw[0])
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
    """GT label -> line in (vx, vy, x0, y0) form."""
    c_px = offset_normalised * image_height
    return (1.0, slope, 0.0, c_px)


def hesse_canonical(vx: float, vy: float, x0: float, y0: float):
    """Canonical Hesse form (nx, ny, rho) with ny >= 0."""
    L = math.hypot(vx, vy)
    nx, ny = -vy / L, vx / L
    if ny < 0 or (ny == 0 and nx < 0):
        nx, ny = -nx, -ny
    rho = nx * x0 + ny * y0
    return nx, ny, rho


def angular_error_deg(line_a, line_b) -> float:
    """Angular distance between two lines, in [0, 90] degrees."""
    theta_a = math.degrees(math.atan2(line_a[1], line_a[0]))
    theta_b = math.degrees(math.atan2(line_b[1], line_b[0]))
    raw = abs(theta_a - theta_b) % 180.0
    return min(raw, 180.0 - raw)


def positional_error_px(line_pred, line_gt) -> float:
    """Absolute delta-rho between predicted and GT line in canonical Hesse form."""
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
    pred_has_horizon: bool
    delta_theta_deg: Optional[float]
    delta_rho_px: Optional[float]
    delta_rho_norm: Optional[float]
    iou: Optional[float]
    latency_ms: float
    failed: bool


def _load_labels(csv_path: Path) -> list[dict]:
    """Read label.csv tolerating both 3-col (legacy) and 4-col schemas."""
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        has_col = "has_horizon" in (reader.fieldnames or [])
        rows = []
        for row in reader:
            hh = (row["has_horizon"].strip().lower() == "true") if has_col else True
            rows.append(
                {
                    "filename": row["filename"],
                    "has_horizon": hh,
                    "slope": float(row["slope"]) if hh and row.get("slope") else None,
                    "offset": float(row["offset"]) if hh and row.get("offset") else None,
                }
            )
    return rows


def evaluate(attempt_dir: Path, dataset_dir: Path, limit: Optional[int] = None, seed: Optional[int] = None):
    detect = load_detector(attempt_dir)
    detect_params = inspect.signature(detect).parameters
    supports_random_seed = "random_seed" in detect_params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in detect_params.values()
    )
    if seed is not None and not supports_random_seed:
        print(f"  WARN: {attempt_dir.name} does not accept random_seed; --seed is ignored", file=sys.stderr)

    labels = _load_labels(dataset_dir / "label.csv")
    if limit is not None:
        labels = labels[:limit]

    total = len(labels)
    results: list[SampleResult] = []
    for i, row in enumerate(labels, 1):
        print(f"\r{C.DIM}  {i}/{total}{C.RESET}", end="", flush=True)
        filename = row["filename"]
        gt_hh = row["has_horizon"]

        img_path = dataset_dir / "images" / filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: could not read {img_path}", file=sys.stderr)
            continue
        H, _W = img.shape[:2]

        t0 = time.perf_counter()
        raw = detect(img, random_seed=seed) if seed is not None and supports_random_seed else detect(img)
        latency_ms = (time.perf_counter() - t0) * 1000

        line_pred, mask_pred, pred_no_horizon = normalise_output(raw)
        if line_pred is None and not pred_no_horizon:
            results.append(SampleResult(filename, gt_hh, True, None, None, None, None, latency_ms, failed=True))
            continue

        pred_hh = not pred_no_horizon

        d_theta = d_rho = d_rho_norm = None
        if gt_hh and pred_hh:
            line_gt = line_from_slope_offset(row["slope"], row["offset"], H)
            d_theta = angular_error_deg(line_pred, line_gt)
            d_rho = positional_error_px(line_pred, line_gt)
            d_rho_norm = d_rho / H

        iou_val: Optional[float] = None
        if mask_pred is not None:
            mask_path = dataset_dir / "masks" / "sky" / (Path(filename).stem + ".png")
            mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
            if mask_gt is not None:
                gt_bool = mask_gt > 127
                pred_bool = mask_pred > 127 if mask_pred.dtype != bool else mask_pred
                iou_val = iou_binary(pred_bool, gt_bool)

        results.append(
            SampleResult(filename, gt_hh, pred_hh, d_theta, d_rho, d_rho_norm, iou_val, latency_ms, failed=False)
        )

    print()
    return results


# --------------------------- reporting --------------------------- #

def _passes_sample(r: SampleResult) -> bool:
    if r.gt_has_horizon != r.pred_has_horizon:
        return False
    if not r.gt_has_horizon:
        return True
    return r.delta_theta_deg < PASS_DTHETA_DEG and r.delta_rho_norm < PASS_DRHO_NORM


def _stats(values: list[float]) -> Optional[dict]:
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "worst": float(arr.max()),
    }


def summarise_results(results: list[SampleResult]) -> dict:
    ok = [r for r in results if not r.failed]
    failed = [r for r in results if r.failed]
    has_no_horizon_labels = any(not r.gt_has_horizon for r in results)

    line_rows = [r for r in ok if r.delta_theta_deg is not None]
    passed = sum(1 for r in ok if _passes_sample(r))
    pass_total = len(results)
    acc_rate = passed / pass_total if pass_total else 0.0
    acc_verdict = "PASS" if acc_rate >= 0.95 else "WARN" if acc_rate >= 0.80 else "FAIL"

    tp = sum(1 for r in ok if r.gt_has_horizon and r.pred_has_horizon)
    fn = sum(1 for r in ok if r.gt_has_horizon and not r.pred_has_horizon)
    fp = sum(1 for r in ok if not r.gt_has_horizon and r.pred_has_horizon)
    tn = sum(1 for r in ok if not r.gt_has_horizon and not r.pred_has_horizon)

    iou_vals = [r.iou for r in ok if r.iou is not None]
    lat = [r.latency_ms for r in ok]
    latency_stats = _stats(lat)
    fps_stats = None
    speed_verdict = "WARN"
    pct_over_budget = None

    if latency_stats is not None:
        fps_stats = {
            "mean": 1000 / latency_stats["mean"],
            "median": 1000 / latency_stats["median"],
            "p90": 1000 / latency_stats["p90"],
            "worst": 1000 / latency_stats["worst"],
        }
        if latency_stats["mean"] <= LATENCY_BUDGET_MS and latency_stats["p90"] <= LATENCY_BUDGET_MS:
            speed_verdict = "PASS"
        elif latency_stats["mean"] <= LATENCY_BUDGET_MS:
            speed_verdict = "WARN"
        else:
            speed_verdict = "FAIL"
        pct_over_budget = sum(1 for v in lat if v > LATENCY_BUDGET_MS) / len(lat) * 100

    return {
        "counts": {
            "total": len(results),
            "evaluated": len(ok),
            "failed": len(failed),
            "line_scored": len(line_rows),
        },
        "has_no_horizon_labels": has_no_horizon_labels,
        "confusion_matrix": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
            "failed_gt_horizon": sum(1 for r in failed if r.gt_has_horizon),
            "failed_gt_no_horizon": sum(1 for r in failed if not r.gt_has_horizon),
        } if has_no_horizon_labels else None,
        "accuracy": {
            "verdict": acc_verdict,
            "passed": passed,
            "total": pass_total,
            "pass_rate": acc_rate,
            "pass_thresholds": {
                "delta_theta_deg_lt": PASS_DTHETA_DEG,
                "delta_rho_norm_lt": PASS_DRHO_NORM,
            },
            "angle_error_deg": _stats([r.delta_theta_deg for r in line_rows]),
            "position_error_norm": _stats([r.delta_rho_norm for r in line_rows]),
            "hesse_distance_px": _stats([r.delta_rho_px for r in line_rows]),
        },
        "mask_iou": _stats(iou_vals),
        "speed": {
            "verdict": speed_verdict,
            "latency_budget_ms": LATENCY_BUDGET_MS,
            "latency_ms": latency_stats,
            "fps": fps_stats,
            "pct_frames_over_budget": pct_over_budget,
        },
        "worst_frames_by_angle": [
            {
                "filename": r.filename,
                "delta_theta_deg": r.delta_theta_deg,
                "delta_rho_norm": r.delta_rho_norm,
                "iou": r.iou,
            }
            for r in sorted(line_rows, key=lambda row: row.delta_theta_deg, reverse=True)[:5]
        ],
    }


def write_full_eval_results(
    results: list[SampleResult],
    attempt_dir: Path,
    dataset_dir: Path,
    limit: Optional[int],
    seed: Optional[int],
    wall_clock_s: float,
) -> Path:
    out_path = attempt_dir / "full-eval-results.json"
    payload = {
        "attempt": attempt_dir.name,
        "attempt_dir": str(attempt_dir),
        "dataset_dir": str(dataset_dir),
        "limit": limit,
        "seed": seed,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_clock_s": wall_clock_s,
        "summary": summarise_results(results),
        "samples": [
            {
                "filename": r.filename,
                "gt_has_horizon": r.gt_has_horizon,
                "pred_has_horizon": r.pred_has_horizon,
                "delta_theta_deg": r.delta_theta_deg,
                "delta_rho_px": r.delta_rho_px,
                "delta_rho_norm": r.delta_rho_norm,
                "iou": r.iou,
                "latency_ms": r.latency_ms,
                "failed": r.failed,
                "passes": False if r.failed else _passes_sample(r),
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {out_path}")
    return out_path

def _stat_row(label, values, fmt, unit="", low_is_good=True):
    """Print one metric row: label  avg  median  9-in-10  worst."""
    arr = np.asarray(values)
    avg = float(arr.mean())
    p50 = float(np.percentile(arr, 50))
    p90 = float(np.percentile(arr, 90))
    vmax = float(arr.max())
    tail_label = "9-in-10 below" if low_is_good else "9-in-10 above"
    print(
        f"  {label:<28s}"
        f"  avg {fmt.format(avg)}{unit}"
        f"   median {fmt.format(p50)}{unit}"
        f"   {tail_label} {fmt.format(p90)}{unit}"
        f"   worst {fmt.format(vmax)}{unit}"
    )


def _verdict(label, passed: bool | None):
    """Print a section header with a coloured PASS / FAIL / WARN badge."""
    if passed is True:
        colour, badge = C.PASS, "PASS"
    elif passed is False:
        colour, badge = C.FAIL, "FAIL"
    else:
        colour, badge = C.WARN, "WARN"
    width = 42
    print(f"\n{C.BOLD}{label:{width}s}{colour}[ {badge} ]{C.RESET}")
    print(C.DIM + "-" * (width + 8) + C.RESET)


def print_report(results: list[SampleResult], attempt_name: str):
    ok = [r for r in results if not r.failed]
    failed = [r for r in results if r.failed]

    print(f"\n{C.BOLD}{'=' * 60}{C.RESET}")
    print(f"  {C.BOLD}{attempt_name}{C.RESET}")
    print(f"  {C.DIM}{len(ok)} frames evaluated" + (f"  ({len(failed)} could not be detected)" if failed else "") + C.RESET)
    print(f"{C.BOLD}{'=' * 60}{C.RESET}")

    if not ok:
        print("\n  No frames detected — nothing to report.")
        return

    has_no_horizon_labels = any(not r.gt_has_horizon for r in results)
    tp = sum(1 for r in ok if r.gt_has_horizon and r.pred_has_horizon)
    fn = sum(1 for r in ok if r.gt_has_horizon and not r.pred_has_horizon)
    fp = sum(1 for r in ok if not r.gt_has_horizon and r.pred_has_horizon)
    tn = sum(1 for r in ok if not r.gt_has_horizon and not r.pred_has_horizon)
    if has_no_horizon_labels:
        failed_gt_horizon = sum(1 for r in failed if r.gt_has_horizon)
        failed_gt_no_horizon = sum(1 for r in failed if not r.gt_has_horizon)
        print("  has_horizon confusion matrix:")
        print(f"    TP (gt=horizon, pred=horizon)         = {tp}")
        print(f"    FN (gt=horizon, pred=no_horizon)      = {fn}")
        print(f"    FP (gt=no_horizon, pred=horizon)      = {fp}")
        print(f"    TN (gt=no_horizon, pred=no_horizon)   = {tn}")
        if failed:
            print(
                f"    failed (no decision)                  = {len(failed)} "
                f"(gt=horizon: {failed_gt_horizon}, gt=no_horizon: {failed_gt_no_horizon})"
            )
        print()

    line_rows = [r for r in ok if r.delta_theta_deg is not None]

    def passes(r: SampleResult) -> bool:
        if r.gt_has_horizon != r.pred_has_horizon:
            return False
        if not r.gt_has_horizon:
            return True
        return r.delta_theta_deg < PASS_DTHETA_DEG and r.delta_rho_norm < PASS_DRHO_NORM

    passed = sum(1 for r in ok if passes(r))
    pass_total = len(results)
    acc_rate = passed / pass_total if pass_total else 0.0
    acc_verdict = True if acc_rate >= 0.95 else None if acc_rate >= 0.80 else False
    _verdict("ACCURACY", acc_verdict)

    rate_colour = C.PASS if acc_verdict is True else C.WARN if acc_verdict is None else C.FAIL
    pass_detail = "correct has_horizon + line within thresholds" if has_no_horizon_labels else "line within thresholds"
    print(
        f"  {rate_colour}{passed}/{pass_total} frames ({acc_rate*100:.1f}%){C.RESET} pass"
        f"  {C.DIM}({pass_detail}; angle < {PASS_DTHETA_DEG:.0f}° and position < {PASS_DRHO_NORM*100:.0f}% of frame height){C.RESET}\n"
    )

    if line_rows:
        _stat_row("Horizon angle error", [r.delta_theta_deg for r in line_rows], "{:.1f}", "°")
        _stat_row("Horizon position error", [r.delta_rho_norm * 100 for r in line_rows], "{:.1f}", "%")
        _stat_row("Hesse distance", [r.delta_rho_px for r in line_rows], "{:.1f}", " px")

    iou_vals = [r.iou for r in ok if r.iou is not None]
    if iou_vals:
        print()
        _stat_row("Sky region overlap", iou_vals, "{:.2f}", "  (1.0 = perfect)")

    lat = [r.latency_ms for r in ok]
    avg_ms = float(np.mean(lat))
    p50_ms = float(np.percentile(lat, 50))
    p90_ms = float(np.percentile(lat, 90))
    max_ms = float(np.max(lat))

    avg_fps = 1000 / avg_ms
    p50_fps = 1000 / p50_ms
    p90_fps = 1000 / p90_ms
    min_fps = 1000 / max_ms

    if avg_ms <= LATENCY_BUDGET_MS and p90_ms <= LATENCY_BUDGET_MS:
        speed_verdict = True
    elif avg_ms <= LATENCY_BUDGET_MS:
        speed_verdict = None
    else:
        speed_verdict = False

    _verdict(f"SPEED  (target: >= 15 FPS  /  <= {LATENCY_BUDGET_MS:.0f} ms per frame)", speed_verdict)
    print(
        f"  {'Per-frame time':<28s}"
        f"  avg {avg_ms:5.1f} ms"
        f"   median {p50_ms:5.1f} ms"
        f"   9-in-10 below {p90_ms:6.1f} ms"
        f"   worst {max_ms:6.1f} ms"
    )
    print(
        f"  {'In FPS':<28s}"
        f"  avg {avg_fps:5.1f}"
        f"   median {p50_fps:5.1f}"
        f"   9-in-10 above {p90_fps:6.1f}"
        f"   worst {min_fps:6.1f}"
    )
    if speed_verdict is None:
        pct_over = sum(1 for v in lat if v > LATENCY_BUDGET_MS) / len(lat) * 100
        print(f"\n  {C.WARN}Note: {pct_over:.0f}% of frames exceed the {LATENCY_BUDGET_MS:.0f} ms budget.{C.RESET}")

    if line_rows:
        print(f"\n\n{C.BOLD}WORST 5 FRAMES{C.RESET}  {C.DIM}(by angle error){C.RESET}")
        print(C.DIM + "-" * 60 + C.RESET)
        for r in sorted(line_rows, key=lambda r: r.delta_theta_deg, reverse=True)[:5]:
            iou_str = f"  sky overlap {r.iou:.0%}" if r.iou is not None else ""
            angle_colour = C.FAIL if r.delta_theta_deg >= PASS_DTHETA_DEG else C.WARN
            print(
                f"  {C.DIM}{r.filename[:48]:48s}{C.RESET}"
                f"  {angle_colour}angle off {r.delta_theta_deg:.1f}°{C.RESET}"
                f"  position off {r.delta_rho_norm*100:.1f}%"
                f"{iou_str}"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("attempt", type=Path, help="Path to an attempt folder, e.g. attempts/attempt-2-rotation-invariant")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to Horizon-UAV dataset root")
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples (for quick iteration)")
    p.add_argument("--seed", type=int, default=None, help="Seed passed to detectors that expose random_seed")
    args = p.parse_args()

    t_start = time.perf_counter()
    results = evaluate(args.attempt, args.dataset, limit=args.limit, seed=args.seed)
    elapsed = time.perf_counter() - t_start

    print_report(results, attempt_name=args.attempt.name)
    write_full_eval_results(results, args.attempt, args.dataset, args.limit, args.seed, elapsed)
    print(f"\n  total wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
