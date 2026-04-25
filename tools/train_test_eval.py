"""
tools/train_test_eval.py — stratified train/test evaluation for a horizon detector.

Usage:
    python tools/train_test_eval.py attempts/attempt-2-rotation-invariant
    python tools/train_test_eval.py attempts/attempt-2-rotation-invariant --seed 7

Evaluates a detector on an 80/20 stratified split of the Horizon-UAV dataset.
Stratification is over angle magnitude (flat / moderate / steep) × horizon
position (low / mid / high), giving up to 9 strata.

Outputs:
  - Console report with train and test metrics side by side.
  - <attempt_dir>/split_results.csv  — per-sample results with split labels.
  - <attempt_dir>/result.md          — "## Train/Test Evaluation" section appended.
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Import shared helpers from evaluate.py (same tools/ directory).
# ---------------------------------------------------------------------------
_TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TOOLS_DIR))
import evaluate as _ev  # noqa: E402

REPO_ROOT = _TOOLS_DIR.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "horizon_uav_dataset"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAP_THRESHOLDS: list[tuple[float, float]] = [
    (1.0, 0.01), (2.0, 0.02), (3.0, 0.03), (5.0, 0.05),
    (7.0, 0.07), (10.0, 0.10), (15.0, 0.15), (20.0, 0.20),
]

WARMUP_FRAMES = 10
DEFAULT_SEED = 42
TEST_FRACTION = 0.20
PASS_DTHETA_DEG = 5.0
PASS_DRHO_NORM = 0.05


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    filename: str
    split: str         # "train" or "test"
    angle_bin: str
    offset_bin: str
    delta_theta_deg: Optional[float]
    delta_rho_px: Optional[float]
    delta_rho_norm: Optional[float]
    iou: Optional[float]
    latency_ms: float
    failed: bool

    @property
    def stratum(self) -> str:
        return f"{self.angle_bin}_{self.offset_bin}"


# ---------------------------------------------------------------------------
# Stratification helpers
# ---------------------------------------------------------------------------

def _angle_bin(slope: float) -> str:
    angle_abs = abs(math.degrees(math.atan(slope)))
    if angle_abs < 15.0:
        return "flat"
    if angle_abs < 45.0:
        return "moderate"
    return "steep"


def _offset_bin(offset: float) -> str:
    if offset < 0.33:
        return "low"
    if offset <= 0.67:
        return "mid"
    return "high"


def stratified_split(
    labels: list[dict],
    test_fraction: float = TEST_FRACTION,
    seed: int = DEFAULT_SEED,
) -> tuple[list[dict], list[dict]]:
    """Return (train_rows, test_rows) as a reproducible stratified split.

    Within each stratum rows are shuffled with `seed`; the first
    ceil(n * test_fraction) shuffled rows go to the test set.
    Any stratum with ≥ 2 samples contributes at least 1 test sample.
    """
    rng = np.random.default_rng(seed)

    strata: dict[str, list] = {}
    for row in labels:
        key = f"{_angle_bin(float(row['slope']))}_{_offset_bin(float(row['offset']))}"
        strata.setdefault(key, []).append(row)

    train_rows: list[dict] = []
    test_rows: list[dict] = []
    for key in sorted(strata):
        rows = strata[key]
        shuffled_idx = rng.permutation(len(rows)).tolist()
        n_test = max(1, math.ceil(len(rows) * test_fraction)) if len(rows) >= 2 else 0
        test_set = set(shuffled_idx[:n_test])
        for i, row in enumerate(rows):
            (test_rows if i in test_set else train_rows).append(row)

    return train_rows, test_rows


# ---------------------------------------------------------------------------
# mAP over a threshold sweep
# ---------------------------------------------------------------------------

def compute_map(results: list[SplitResult]) -> tuple[float, list[float]]:
    """Mean precision over MAP_THRESHOLDS.

    At each threshold (Δθ_max, Δρ/H_max), precision = fraction of
    non-failed frames where both errors are within that threshold.
    mAP = mean(precisions).
    """
    ok = [r for r in results if not r.failed and r.delta_theta_deg is not None]
    if not ok:
        return 0.0, [0.0] * len(MAP_THRESHOLDS)
    precisions = []
    for dtheta_max, drho_max in MAP_THRESHOLDS:
        passing = sum(
            1 for r in ok
            if r.delta_theta_deg <= dtheta_max and r.delta_rho_norm <= drho_max
        )
        precisions.append(passing / len(ok))
    return float(np.mean(precisions)), precisions


# ---------------------------------------------------------------------------
# Speed helper
# ---------------------------------------------------------------------------

def _fps(latencies_ms: list[float], warmup: int = 0) -> Optional[float]:
    valid = latencies_ms[warmup:]
    if not valid:
        return None
    total_s = sum(valid) / 1000.0
    return len(valid) / total_s if total_s > 0 else None


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_split_eval(
    attempt_dir: Path,
    dataset_dir: Path,
    seed: int = DEFAULT_SEED,
) -> tuple[list[SplitResult], list[SplitResult]]:
    """Evaluate a detector with a stratified 80/20 split.

    Training frames run first so the warm-up frames come from the training set.
    Returns (train_results, test_results).
    """
    detect = _ev.load_detector(attempt_dir)

    with (dataset_dir / "label.csv").open() as f:
        all_labels = list(csv.DictReader(f))

    train_rows, test_rows = stratified_split(all_labels, TEST_FRACTION, seed)
    ordered = [("train", row) for row in train_rows] + [("test", row) for row in test_rows]
    total = len(ordered)

    all_results: list[SplitResult] = []
    for i, (split, row) in enumerate(ordered, 1):
        print(f"\r  {i}/{total}", end="", flush=True)

        filename = row["filename"]
        slope = float(row["slope"])
        offset = float(row["offset"])

        img_path = dataset_dir / "images" / filename
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"\n  WARN: could not read {img_path}", file=sys.stderr)
            continue

        H = img.shape[0]

        t0 = time.perf_counter()
        raw = detect(img)
        latency_ms = (time.perf_counter() - t0) * 1000

        line_pred, mask_pred, _pred_no_horizon = _ev.normalise_output(raw)
        ab = _angle_bin(slope)
        ob = _offset_bin(offset)

        if line_pred is None:
            all_results.append(SplitResult(
                filename, split, ab, ob,
                None, None, None, None, latency_ms, failed=True,
            ))
            continue

        line_gt = _ev.line_from_slope_offset(slope, offset, H)
        d_theta = _ev.angular_error_deg(line_pred, line_gt)
        d_rho = _ev.positional_error_px(line_pred, line_gt)
        d_rho_norm = d_rho / H

        iou_val: Optional[float] = None
        if mask_pred is not None:
            mask_path = dataset_dir / "masks" / "sky" / (Path(filename).stem + ".png")
            mask_gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
            if mask_gt is not None:
                gt_bool = mask_gt > 127
                pred_bool = mask_pred > 127 if mask_pred.dtype != bool else mask_pred
                iou_val = _ev.iou_binary(pred_bool, gt_bool)

        all_results.append(SplitResult(
            filename, split, ab, ob,
            d_theta, d_rho, d_rho_norm, iou_val, latency_ms, failed=False,
        ))

    print()

    train_results = [r for r in all_results if r.split == "train"]
    test_results = [r for r in all_results if r.split == "test"]
    return train_results, test_results


# ---------------------------------------------------------------------------
# Metrics aggregation
# ---------------------------------------------------------------------------

def _aggregate(results: list[SplitResult], warmup: int = 0) -> dict:
    ok = [r for r in results if not r.failed and r.delta_theta_deg is not None]
    latencies = [r.latency_ms for r in results]

    n_total = len(results)
    n_ok = len(ok)

    fps = _fps(latencies, warmup=warmup)
    warm_lats = latencies[warmup:]
    mean_lat = float(np.mean(warm_lats)) if warm_lats else None

    base = {
        "n_total": n_total,
        "n_ok": n_ok,
        "n_failed": n_total - n_ok,
        "fps": fps,
        "mean_lat_ms": mean_lat,
        "mean_theta": None,
        "p50_theta": None,
        "p90_theta": None,
        "mean_rho_pct": None,
        "p50_rho_pct": None,
        "p90_rho_pct": None,
        "mean_iou": None,
        "pass_rate": None,
        "map": 0.0,
        "precisions": [0.0] * len(MAP_THRESHOLDS),
    }

    if not ok:
        return base

    angles = np.asarray([r.delta_theta_deg for r in ok], dtype=float)
    rhos = np.asarray([r.delta_rho_norm for r in ok], dtype=float)
    ious = [r.iou for r in ok if r.iou is not None]

    pass_count = sum(
        1 for r in ok
        if r.delta_theta_deg < PASS_DTHETA_DEG and r.delta_rho_norm < PASS_DRHO_NORM
    )

    map_val, precisions = compute_map(results)

    return {
        **base,
        "mean_theta": float(angles.mean()),
        "p50_theta": float(np.percentile(angles, 50)),
        "p90_theta": float(np.percentile(angles, 90)),
        "mean_rho_pct": float(rhos.mean()) * 100,
        "p50_rho_pct": float(np.percentile(rhos, 50)) * 100,
        "p90_rho_pct": float(np.percentile(rhos, 90)) * 100,
        "mean_iou": float(np.mean(ious)) if ious else None,
        "pass_rate": pass_count / n_ok,
        "map": map_val,
        "precisions": precisions,
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _fmt(v: Optional[float], spec: str = ".2f") -> str:
    return format(v, spec) if v is not None else "N/A"


def print_report(
    train_results: list[SplitResult],
    test_results: list[SplitResult],
    attempt_name: str,
    seed: int,
) -> None:
    tr = _aggregate(train_results, warmup=WARMUP_FRAMES)
    te = _aggregate(test_results, warmup=0)

    def row(label: str, tv: str, ev: str) -> None:
        print(f"  {label:<32s}  {tv:>10s}  {ev:>10s}")

    print(f"\n{'=' * 60}")
    print(f"  {attempt_name}  (seed={seed})")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<32s}  {'Train':>10s}  {'Test':>10s}")
    print(f"  {'-' * 56}")

    n_tr = f"{tr['n_ok']}" + (f" (-{tr['n_failed']})" if tr["n_failed"] else "")
    n_te = f"{te['n_ok']}" + (f" (-{te['n_failed']})" if te["n_failed"] else "")
    row("N evaluated", n_tr, n_te)
    row(f"FPS (excl. {WARMUP_FRAMES}-frame warmup)", _fmt(tr["fps"], ".1f"), _fmt(te["fps"], ".1f"))
    row("Mean latency (ms)", _fmt(tr["mean_lat_ms"]), _fmt(te["mean_lat_ms"]))

    print(f"  {'-' * 56}")

    row("Mean angle error (°)", _fmt(tr["mean_theta"]), _fmt(te["mean_theta"]))
    row("P90 angle error (°)", _fmt(tr["p90_theta"]), _fmt(te["p90_theta"]))
    row("Mean position error (%H)", _fmt(tr["mean_rho_pct"]), _fmt(te["mean_rho_pct"]))
    row("P90 position error (%H)", _fmt(tr["p90_rho_pct"]), _fmt(te["p90_rho_pct"]))
    row("Mean IoU", _fmt(tr["mean_iou"], ".3f"), _fmt(te["mean_iou"], ".3f"))

    print(f"  {'-' * 56}")

    pass_tr = f"{tr['pass_rate'] * 100:.1f}%" if tr["pass_rate"] is not None else "N/A"
    pass_te = f"{te['pass_rate'] * 100:.1f}%" if te["pass_rate"] is not None else "N/A"
    row("Pass rate (5°/5%H)", pass_tr, pass_te)
    row("mAP (threshold sweep)", _fmt(tr["map"], ".4f"), _fmt(te["map"], ".4f"))

    print(f"\n  mAP threshold breakdown:")
    for (dtheta, drho), p_tr, p_te in zip(MAP_THRESHOLDS, tr["precisions"], te["precisions"]):
        print(
            f"    Δθ < {dtheta:4.0f}°  Δρ < {drho * 100:3.0f}%H"
            f"  →  train {p_tr:.3f}  test {p_te:.3f}"
        )

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(
    train_results: list[SplitResult],
    test_results: list[SplitResult],
    attempt_dir: Path,
) -> None:
    out_path = attempt_dir / "split_results.csv"
    fieldnames = [
        "filename", "split", "angle_bin", "offset_bin", "stratum",
        "delta_theta_deg", "delta_rho_px", "delta_rho_norm",
        "iou", "latency_ms", "failed",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in train_results + test_results:
            writer.writerow({
                "filename": r.filename,
                "split": r.split,
                "angle_bin": r.angle_bin,
                "offset_bin": r.offset_bin,
                "stratum": r.stratum,
                "delta_theta_deg": f"{r.delta_theta_deg:.4f}" if r.delta_theta_deg is not None else "",
                "delta_rho_px": f"{r.delta_rho_px:.4f}" if r.delta_rho_px is not None else "",
                "delta_rho_norm": f"{r.delta_rho_norm:.6f}" if r.delta_rho_norm is not None else "",
                "iou": f"{r.iou:.4f}" if r.iou is not None else "",
                "latency_ms": f"{r.latency_ms:.3f}",
                "failed": int(r.failed),
            })
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# result.md output
# ---------------------------------------------------------------------------

def write_result_md(
    train_results: list[SplitResult],
    test_results: list[SplitResult],
    attempt_dir: Path,
    seed: int,
) -> None:
    tr = _aggregate(train_results, warmup=WARMUP_FRAMES)
    te = _aggregate(test_results, warmup=0)

    def fv(v: Optional[float], spec: str = ".2f") -> str:
        return _fmt(v, spec)

    pass_tr = f"{tr['pass_rate'] * 100:.1f}%" if tr["pass_rate"] is not None else "N/A"
    pass_te = f"{te['pass_rate'] * 100:.1f}%" if te["pass_rate"] is not None else "N/A"

    lines: list[str] = [
        "",
        "## Train/Test Evaluation",
        "",
        f"seed={seed} | train={tr['n_total']} | test={te['n_total']}",
        "",
        "| Metric | Train | Test |",
        "|---|---|---|",
        f"| N evaluated | {tr['n_ok']} | {te['n_ok']} |",
        f"| N failed | {tr['n_failed']} | {te['n_failed']} |",
        f"| FPS (excl. {WARMUP_FRAMES}-frame warmup) | {fv(tr['fps'], '.1f')} | {fv(te['fps'], '.1f')} |",
        f"| Mean latency (ms) | {fv(tr['mean_lat_ms'])} | {fv(te['mean_lat_ms'])} |",
        f"| Mean angle error (°) | {fv(tr['mean_theta'])} | {fv(te['mean_theta'])} |",
        f"| P90 angle error (°) | {fv(tr['p90_theta'])} | {fv(te['p90_theta'])} |",
        f"| Mean position error (%H) | {fv(tr['mean_rho_pct'])} | {fv(te['mean_rho_pct'])} |",
        f"| P90 position error (%H) | {fv(tr['p90_rho_pct'])} | {fv(te['p90_rho_pct'])} |",
        f"| Mean IoU | {fv(tr['mean_iou'], '.3f')} | {fv(te['mean_iou'], '.3f')} |",
        f"| Pass rate (Δθ<5° & Δρ<5%H) | {pass_tr} | {pass_te} |",
        f"| mAP (threshold sweep) | {fv(tr['map'], '.4f')} | {fv(te['map'], '.4f')} |",
        "",
        "**mAP threshold breakdown:**",
        "",
        "| Δθ max | Δρ/H max | Train precision | Test precision |",
        "|---|---|---|---|",
    ]

    for (dtheta, drho), p_tr, p_te in zip(MAP_THRESHOLDS, tr["precisions"], te["precisions"]):
        lines.append(f"| {dtheta:.0f}° | {drho * 100:.0f}% | {p_tr:.3f} | {p_te:.3f} |")

    lines.append("")

    result_md = attempt_dir / "result.md"
    existing = result_md.read_text() if result_md.exists() else ""

    marker = "## Train/Test Evaluation"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip()

    result_md.write_text(existing + "\n".join(lines) + "\n")
    print(f"  wrote {result_md}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("attempt", type=Path, help="Path to an attempt folder")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed for the split")
    args = p.parse_args()

    t_wall = time.perf_counter()
    train_results, test_results = run_split_eval(args.attempt, args.dataset, seed=args.seed)
    elapsed = time.perf_counter() - t_wall

    print_report(train_results, test_results, attempt_name=args.attempt.name, seed=args.seed)
    write_csv(train_results, test_results, args.attempt)
    write_result_md(train_results, test_results, args.attempt, seed=args.seed)
    print(f"  total wall-clock time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
