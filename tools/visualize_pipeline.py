"""
tools/visualize_pipeline.py — render attempt 8's pipeline as labelled stage images.

Re-uses attempt-8's internal primitives (_extract_boundary, _ransac_topk,
_ettinger_score) so the visualization is exactly what the detector does, not a
re-implementation that could drift. Output is a directory of PNGs, one per
pipeline stage, with a title strip at the top — suitable for dropping into a
slide deck.

Usage:
    .venv/bin/python tools/visualize_pipeline.py <image-path> --out <out-dir>
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTEMPT_DIR = REPO_ROOT / "attempts" / "attempt-8-temporal-prior"


def load_attempt_module():
    spec = importlib.util.spec_from_file_location("attempt8", ATTEMPT_DIR / "horizon_detect.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["attempt8"] = module
    spec.loader.exec_module(module)
    return module


def add_title(img: np.ndarray, title: str, subtitle: str | None = None) -> np.ndarray:
    """Add a black header strip with title + optional subtitle above the image."""
    h, w = img.shape[:2]
    strip_h = 70 if subtitle else 44
    canvas = np.zeros((h + strip_h, w, 3), dtype=np.uint8)
    canvas[strip_h:, :, :] = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(canvas, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    return canvas


def colorize_channel(channel: np.ndarray) -> np.ndarray:
    """Stretch a single-channel image to BGR for consistent saving."""
    if channel.dtype != np.uint8:
        lo, hi = float(channel.min()), float(channel.max())
        scaled = ((channel - lo) / max(hi - lo, 1e-6) * 255).clip(0, 255).astype(np.uint8)
    else:
        scaled = channel
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


def line_endpoints(line, shape):
    vx, vy, x0, y0 = (float(v) for v in line)
    h, w = shape[:2]
    s = max(h, w) * 2
    return (int(round(x0 - s * vx)), int(round(y0 - s * vy))), (int(round(x0 + s * vx)), int(round(y0 + s * vy)))


def draw_line(canvas: np.ndarray, line, color, thickness=2):
    p1, p2 = line_endpoints(line, canvas.shape)
    cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)


def overlay_boundary(image_bgr: np.ndarray, boundary: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    out = image_bgr.copy()
    out[boundary > 0] = color
    return out


def render_pipeline(image_path: Path, out_dir: Path, attempt) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"could not read {image_path}")
    h, w = img.shape[:2]

    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1 — input ---
    cv2.imwrite(str(out_dir / "01_input.png"),
                add_title(img, "1. Input frame", f"{image_path.name}  ({w}x{h})"))

    # --- Stage 2 — Lab decomposition ---
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    L, a_star, b_star = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    cv2.imwrite(str(out_dir / "02a_lab_L.png"),
                add_title(colorize_channel(L), "2a. Lab L (luminance)",
                          "How bright is each pixel? Sees haze and overcast like grayscale."))
    cv2.imwrite(str(out_dir / "02b_lab_b.png"),
                add_title(colorize_channel(b_star), "2b. Lab b* (blue-yellow)",
                          "Blue sky -> dark, yellow/green ground -> bright. Survives glare and overcast."))

    # --- Stage 3 — Otsu masks (per channel) ---
    boundary_L, mask_L, balance_L = attempt._extract_boundary(L)
    boundary_b, mask_b, balance_b = attempt._extract_boundary(b_star)
    cv2.imwrite(str(out_dir / "03a_otsu_L.png"),
                add_title(colorize_channel(mask_L), "3a. Otsu split on Lab L",
                          f"sky/ground threshold from luminance.  larger-class fraction = {balance_L:.2f}"))
    cv2.imwrite(str(out_dir / "03b_otsu_b.png"),
                add_title(colorize_channel(mask_b), "3b. Otsu split on Lab b*",
                          f"sky/ground threshold from colour.  larger-class fraction = {balance_b:.2f}"))

    # --- Stage 4 — boundary after gradient orientation filter ---
    cv2.imwrite(str(out_dir / "04a_boundary_L.png"),
                add_title(overlay_boundary(img, boundary_L, (0, 255, 0)),
                          "4a. Filtered boundary - L channel",
                          "Sobel orientation filter discards near-vertical edges (tree trunks, frame borders)."))
    cv2.imwrite(str(out_dir / "04b_boundary_b.png"),
                add_title(overlay_boundary(img, boundary_b, (0, 255, 255)),
                          "4b. Filtered boundary - b* channel",
                          "Same filter on the colour-derived mask gives a different set of boundary points."))

    # --- Stage 5 — pooled RANSAC candidates from both channels ---
    rng = np.random.default_rng(0)
    pooled = []  # (count, line, channel_label)
    for label, boundary, color in (("L", boundary_L, (0, 255, 0)), ("b*", boundary_b, (0, 255, 255))):
        ys, xs = np.where(boundary > 0)
        if len(xs) < attempt._MIN_BOUNDARY_PTS:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) > attempt._MAX_BOUNDARY_PTS:
            idx = rng.choice(len(pts), attempt._MAX_BOUNDARY_PTS, replace=False)
            pts = pts[idx]
        for count, inlier_mask, seed in attempt._ransac_topk(
            pts, attempt._RANSAC_ITER, attempt._INLIER_THRESHOLD, attempt._TOP_K, rng
        ):
            pooled.append((count, seed, label, color, pts, inlier_mask))

    cands_canvas = img.copy()
    for _count, seed, label, color, _pts, _inl in pooled:
        draw_line(cands_canvas, seed, color, thickness=1)
    cv2.putText(cands_canvas, f"{len(pooled)} candidate lines (green=L, yellow=b*)",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "05_ransac_candidates.png"),
                add_title(cands_canvas, "5. Pooled RANSAC candidates",
                          "Top-K hypotheses per channel. Pick by inlier count = wrong on tricky frames; use coherence next."))

    # --- Stage 6 — Ettinger coherence rerank ---
    lab_thumb = cv2.resize(lab, (attempt._RERANK_THUMB, attempt._RERANK_THUMB), interpolation=cv2.INTER_AREA)
    scored = []  # (raw_score, candidate_index)
    for i, (_count, seed, _label, _color, _pts, _inl) in enumerate(pooled):
        score = attempt._ettinger_score(seed, lab_thumb, w, h)
        scored.append((score, i))
    scored.sort(reverse=True)

    rerank_canvas = img.copy()
    # draw runners-up faint
    for score, idx in scored[1:]:
        _count, seed, _label, _color, _pts, _inl = pooled[idx]
        draw_line(rerank_canvas, seed, (110, 110, 110), thickness=1)
    # draw winner
    if scored:
        winner_score, winner_idx = scored[0]
        _count, win_seed, win_label, _color, _pts, _inl = pooled[winner_idx]
        draw_line(rerank_canvas, win_seed, (0, 0, 255), thickness=3)
        cv2.putText(
            rerank_canvas,
            f"winner: channel={win_label}  coherence={winner_score:.2f}",
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )
    cv2.imwrite(str(out_dir / "06_ettinger_rerank.png"),
                add_title(rerank_canvas, "6. Ettinger coherence rerank",
                          "Score: between-region colour separation / within-region scatter. Picks the line that physically separates two coherent regions."))

    # --- Stage 7 — final Huber refit ---
    result = attempt.detect_horizon(img, random_seed=0)
    final_canvas = attempt._draw(img, result)
    if isinstance(result, dict):
        sub = (
            f"angle={result['angle_deg']:+.2f} deg   "
            f"offset={result['intercept_y_at_x0']:+.1f} px   "
            f"coherence={result['coherence']:.2f}"
        )
    else:
        sub = "no horizon"
    cv2.imwrite(str(out_dir / "07_final.png"),
                add_title(final_canvas, "7. Final horizon (Huber refit on winner inliers)", sub))

    print(f"wrote 9 stage images to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    attempt = load_attempt_module()
    render_pipeline(args.image, args.out, attempt)


if __name__ == "__main__":
    main()
