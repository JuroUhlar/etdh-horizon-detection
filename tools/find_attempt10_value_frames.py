"""
tools/find_attempt10_value_frames.py — find frames where attempt 10 diverges
from attempt 8 and the divergence improves accuracy against ground truth.

Why: attempt 10 inherits the L+b* RANSAC pool of attempt 8 unchanged; if the
two final lines disagree on a frame, the winner must have come from one of
attempt 10's added candidates (DP boundary or top-connected sky envelope).
That's a "value-add" frame — pedagogically the right hero for a slide that
explains what attempt 10 contributes.

Output: top-N frames sorted by |Δθ(attempt8 vs attempt10)|, with each row
showing both attempts' angle/offset, the ground-truth angle/offset, and which
attempt is closer in Hesse normal form.

Usage:
    .venv/bin/python tools/find_attempt10_value_frames.py
    .venv/bin/python tools/find_attempt10_value_frames.py --dataset data/video_clips_fpv_atv --topn 20
"""

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_attempt(name: str, attempt_dir: Path):
    spec = importlib.util.spec_from_file_location(name, attempt_dir / "horizon_detect.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def line_to_hesse(line):
    """(vx, vy, x0, y0) -> (theta_rad, rho_px). Hesse: x cos t + y sin t = rho."""
    vx, vy, x0, y0 = (float(v) for v in line)
    nx, ny = -vy, vx
    norm = math.hypot(nx, ny)
    nx, ny = nx / norm, ny / norm
    rho = nx * x0 + ny * y0
    if rho < 0:
        nx, ny, rho = -nx, -ny, -rho
    theta = math.atan2(ny, nx)
    return theta, rho


def angle_delta_deg(a_rad: float, b_rad: float) -> float:
    """Smallest unsigned angle between two undirected lines, in degrees."""
    d = abs(a_rad - b_rad) % math.pi
    return math.degrees(min(d, math.pi - d))


def line_from_dict(raw):
    if isinstance(raw, dict) and raw.get("line") is not None:
        return tuple(raw["line"])
    return None


def line_from_slope_offset(slope: float, offset_normalised: float, image_height: int):
    """Match tools/evaluate.py: GT (slope, normalised offset) -> (vx, vy, x0, y0)."""
    c_px = offset_normalised * image_height
    return (1.0, slope, 0.0, c_px)


def load_groundtruth(dataset_dir: Path) -> dict:
    """Map filename -> (slope, offset_normalised, has_horizon)."""
    label_csv = dataset_dir / "label.csv"
    if not label_csv.exists():
        return {}
    out = {}
    with label_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("filename")
            if name is None:
                continue
            has_h_raw = row.get("has_horizon")
            has_h = (has_h_raw is None) or (str(has_h_raw).strip().lower() == "true")
            try:
                slope = float(row["slope"]) if row.get("slope") not in (None, "") else None
                off = float(row["offset"]) if row.get("offset") not in (None, "") else None
            except (KeyError, ValueError):
                slope, off = None, None
            out[name] = (slope, off, has_h)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=REPO_ROOT / "data" / "horizon_uav_dataset")
    parser.add_argument("--topn", type=int, default=15)
    parser.add_argument("--min-delta-deg", type=float, default=2.0,
                        help="Skip frames where the two attempts' angles differ by less than this.")
    args = parser.parse_args()

    a8 = load_attempt("a8mod", REPO_ROOT / "attempts" / "attempt-8-temporal-prior")
    a10 = load_attempt("a10mod", REPO_ROOT / "attempts" / "attempt-10-top-connected-sky")

    images_dir = args.dataset / "images"
    if not images_dir.exists():
        raise SystemExit(f"images dir missing: {images_dir}")

    gt = load_groundtruth(args.dataset)
    print(f"loaded {len(gt)} ground-truth labels from {args.dataset / 'label.csv'}")

    rows = []
    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for i, path in enumerate(image_paths, 1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]

        r8 = a8.detect_horizon(img, random_seed=0)
        r10 = a10.detect_horizon(img, random_seed=0)
        l8 = line_from_dict(r8) if isinstance(r8, dict) else None
        l10 = line_from_dict(r10) if isinstance(r10, dict) else None
        if l8 is None or l10 is None:
            continue

        t8, p8 = line_to_hesse(l8)
        t10, p10 = line_to_hesse(l10)
        d_theta = angle_delta_deg(t8, t10)
        if d_theta < args.min_delta_deg:
            continue

        # Compare to ground truth where available.
        gt_entry = gt.get(path.name)
        gt_pass8 = gt_pass10 = closer = None
        if gt_entry is not None and gt_entry[0] is not None and gt_entry[2]:
            slope_gt, off_gt, _has_h = gt_entry
            gt_line = line_from_slope_offset(slope_gt, off_gt, h)
            tg, pg = line_to_hesse(gt_line)
            d8_theta = angle_delta_deg(t8, tg)
            d10_theta = angle_delta_deg(t10, tg)
            d8_rho = abs(p8 - pg) / h
            d10_rho = abs(p10 - pg) / h
            gt_pass8 = (d8_theta < 5.0) and (d8_rho < 0.05)
            gt_pass10 = (d10_theta < 5.0) and (d10_rho < 0.05)
            closer = "a10" if d10_theta < d8_theta else ("a8" if d8_theta < d10_theta else "tie")

        rows.append({
            "path": path,
            "h": h,
            "d_theta": d_theta,
            "a8_theta_deg": math.degrees(t8),
            "a10_theta_deg": math.degrees(t10),
            "a8_rho": p8,
            "a10_rho": p10,
            "gt_pass8": gt_pass8,
            "gt_pass10": gt_pass10,
            "closer": closer,
        })
        if i % 50 == 0:
            print(f"  scanned {i}/{len(image_paths)}, kept {len(rows)} divergent frames")

    rows.sort(key=lambda r: r["d_theta"], reverse=True)

    # Bucket A: frames where a10 passes the GT gate AND a8 fails. Strongest signal.
    bucket_a = [r for r in rows if r["gt_pass10"] is True and r["gt_pass8"] is False]
    # Bucket B: a10 closer to GT than a8 by angle (even if both pass or both fail).
    bucket_b = [r for r in rows if r["closer"] == "a10" and r not in bucket_a]
    # Bucket C: any large divergence, in case GT is unavailable.
    bucket_c = [r for r in rows if r not in bucket_a and r not in bucket_b]

    def fmt(r, idx):
        name = r["path"].name
        line1 = f"  [{idx}] Δθ(a8,a10)={r['d_theta']:5.2f}°   {name[:78]}"
        line2 = (f"      a8 θ={r['a8_theta_deg']:+6.2f}° ρ={r['a8_rho']:+7.1f}    "
                 f"a10 θ={r['a10_theta_deg']:+6.2f}° ρ={r['a10_rho']:+7.1f}    "
                 f"GT-pass a8/a10={r['gt_pass8']}/{r['gt_pass10']}  closer={r['closer']}")
        return line1 + "\n" + line2

    for label, bucket in (("A: a10 PASS, a8 FAIL", bucket_a),
                          ("B: a10 closer to GT than a8", bucket_b),
                          ("C: divergent (GT unknown / both wrong)", bucket_c)):
        print()
        print(f"== bucket {label} — {len(bucket)} frames ==")
        for idx, r in enumerate(bucket[:args.topn]):
            print(fmt(r, idx))


if __name__ == "__main__":
    main()
