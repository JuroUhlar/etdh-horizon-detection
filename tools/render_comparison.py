"""
tools/render_comparison.py — render an attempt-8 vs attempt-10 comparison image.

Loads both attempts, runs each on a single image, and produces a labelled
side-by-side panel: attempt 8's line on the left, attempt 10's line on the
right, with the ground-truth line (from label.csv, if available) overlaid in
both panels as a dashed reference.

This is the "value-add" slide content: a frame where attempt 10's added
candidate sources (DP boundary or top-connected sky envelope) override the
RANSAC winner that attempt 8 would have shipped.

Usage:
    .venv/bin/python tools/render_comparison.py <image-path> --out <out-png> \
        [--dataset data/video_clips_fpv_atv]
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


def line_endpoints(line, shape):
    vx, vy, x0, y0 = (float(v) for v in line)
    h, w = shape[:2]
    s = max(h, w) * 2
    return (int(round(x0 - s * vx)), int(round(y0 - s * vy))), (int(round(x0 + s * vx)), int(round(y0 + s * vy)))


def draw_line(canvas, line, color, thickness=3):
    p1, p2 = line_endpoints(line, canvas.shape)
    cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)


def draw_dashed_line(canvas, line, color, dash_px=14, gap_px=10, thickness=2):
    p1, p2 = line_endpoints(line, canvas.shape)
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    dx, dy = (x2 - x1) / dist, (y2 - y1) / dist
    segment = dash_px + gap_px
    n = int(dist // segment) + 1
    for i in range(n):
        sa = i * segment
        sb = min(sa + dash_px, dist)
        a = (int(x1 + dx * sa), int(y1 + dy * sa))
        b = (int(x1 + dx * sb), int(y1 + dy * sb))
        cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)


def label_panel(panel, header, sub, ok: bool | None):
    """Stamp a title + subtitle bar on the bottom of a panel.

    cv2.putText only renders the Hershey ASCII glyph set, so the caller is
    responsible for keeping `header` and `sub` ASCII-only — no Δ, no degree
    symbol, no superscripts. Using anything else silently renders as '?'.
    """
    h, w = panel.shape[:2]
    bar_h = 96
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    panel[h - bar_h:, :, :] = cv2.addWeighted(panel[h - bar_h:, :, :], 0.25, bar, 0.75, 0)

    if ok is True:
        color_strip = (60, 200, 60)
    elif ok is False:
        color_strip = (60, 60, 230)
    else:
        color_strip = (200, 200, 200)
    cv2.rectangle(panel, (0, h - bar_h), (12, h), color_strip, -1)

    cv2.putText(panel, header, (24, h - bar_h + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, sub, (24, h - bar_h + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 240, 255), 2, cv2.LINE_AA)


def load_groundtruth(dataset_dir: Path):
    label_csv = dataset_dir / "label.csv"
    if not label_csv.exists():
        return {}
    out = {}
    with label_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out[row["filename"]] = (float(row["slope"]), float(row["offset"]))
            except (KeyError, ValueError):
                continue
    return out


def line_from_slope_offset(slope, offset_norm, h):
    return (1.0, slope, 0.0, offset_norm * h)


def angle_deg_of(line):
    """Roll angle in degrees, in (-90, 90]."""
    vx, vy, *_ = line
    return math.degrees(math.atan2(float(vy), float(vx)))


def render_comparison(image_path: Path, out_path: Path, dataset_dir: Path | None):
    a8 = load_attempt("a8mod", REPO_ROOT / "attempts" / "attempt-8-temporal-prior")
    a10 = load_attempt("a10mod", REPO_ROOT / "attempts" / "attempt-10-top-connected-sky")

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    h, w = img.shape[:2]

    r8 = a8.detect_horizon(img, random_seed=0)
    r10 = a10.detect_horizon(img, random_seed=0)
    line8 = r8["line"] if isinstance(r8, dict) else None
    line10 = r10["line"] if isinstance(r10, dict) else None

    gt_line = None
    gt_angle = None
    gt_off = None
    if dataset_dir is not None:
        gt_map = load_groundtruth(dataset_dir)
        if image_path.name in gt_map:
            slope, off = gt_map[image_path.name]
            gt_line = line_from_slope_offset(slope, off, h)
            gt_angle = math.degrees(math.atan(slope))
            gt_off = off * h

    panels = []
    for label, raw_line, color in (("attempt 8", line8, (60, 60, 230)),
                                    ("attempt 10", line10, (60, 200, 60))):
        canvas = img.copy()
        if gt_line is not None:
            draw_dashed_line(canvas, gt_line, (255, 255, 255), thickness=2)
        if raw_line is not None:
            draw_line(canvas, raw_line, color, thickness=4)

        if raw_line is not None:
            ang = angle_deg_of(raw_line)
            if gt_angle is not None:
                d_theta = abs(((ang - gt_angle) + 90) % 180 - 90)
                ok = bool(d_theta < 5.0)
                verdict = "PASS" if ok else "FAIL"
                sub = f"angle={ang:+.1f} deg   GT={gt_angle:+.1f} deg   d_theta={d_theta:.1f} deg   -> {verdict}"
            else:
                ok = None
                sub = f"angle={ang:+.1f} deg"
            label_panel(canvas, label, sub, ok)
        else:
            label_panel(canvas, label, "no line returned", None)
        panels.append(canvas)

    # Side-by-side composite with a separator
    sep = np.full((h, 4, 3), 30, dtype=np.uint8)
    composite = np.hstack([panels[0], sep, panels[1]])

    # Top header strip
    header_h = 78
    header = np.zeros((header_h, composite.shape[1], 3), dtype=np.uint8)
    header_text = "Attempt 8  vs.  attempt 10  -  same frame, same input  (white dashed = ground truth)"
    cv2.putText(header, header_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    full = np.vstack([header, composite])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), full)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=None,
                        help="Dataset root (with label.csv) — used to overlay GT line")
    args = parser.parse_args()
    render_comparison(args.image, args.out, args.dataset)


if __name__ == "__main__":
    main()
