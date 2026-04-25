"""
tools/render_outputs.py — batch-render visual outputs for an attempt.

This runs an attempt over a directory of input images, writes annotated frames
under outputs/, and stitches those frames into an MP4 so the run is easy to
review visually.

Usage:
    .venv/bin/python tools/render_outputs.py attempts/attempt-2-rotation-invariant --images data/samples
    .venv/bin/python tools/render_outputs.py attempts/attempt-3-top-n-ransac --images data/horizon_uav_dataset/images --limit 100
"""

import argparse
import importlib.util
import math
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from stitch_video import DEFAULT_FRAME_DURATION_S, stitch_frames_to_video

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGES = REPO_ROOT / "data" / "horizon_uav_dataset" / "images"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_attempt_module(attempt_dir: Path):
    script = attempt_dir / "horizon_detect.py"
    if not script.exists():
        raise SystemExit(f"No horizon_detect.py in {attempt_dir}")

    spec = importlib.util.spec_from_file_location(f"attempt_{attempt_dir.name}", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if not hasattr(module, "detect_horizon"):
        raise SystemExit(f"{script} does not expose detect_horizon()")
    return module


def discover_images(images_dir: Path, limit: int | None) -> list[Path]:
    if not images_dir.exists():
        raise SystemExit(f"Image directory does not exist: {images_dir}")

    paths = [
        path for path in sorted(images_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No images found in: {images_dir}")
    return paths


def source_name(images_dir: Path) -> str:
    if images_dir.name.lower() in {"images", "frames"} and images_dir.parent != images_dir:
        return images_dir.parent.name
    return images_dir.name


def line_endpoints(line, image_shape) -> tuple[tuple[int, int], tuple[int, int]]:
    vx, vy, x0, y0 = [float(value) for value in line]
    height, width = image_shape[:2]
    scale = max(height, width) * 2
    p1 = (int(round(x0 - scale * vx)), int(round(y0 - scale * vy)))
    p2 = (int(round(x0 + scale * vx)), int(round(y0 + scale * vy)))
    return p1, p2


def overlay_label(image_bgr: np.ndarray, text: str, y: int = 25, color=(0, 255, 255)) -> None:
    cv2.putText(image_bgr, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def fallback_draw(image_bgr: np.ndarray, raw) -> np.ndarray:
    out = image_bgr.copy()

    if raw is None:
        overlay_label(out, "no horizon detected", color=(0, 0, 255))
        return out

    if isinstance(raw, tuple) and len(raw) == 3:
        slope_deg, intercept_px, _mask = raw
        slope = math.tan(math.radians(float(slope_deg)))
        line = (1.0, slope, 0.0, float(intercept_px))
        p1, p2 = line_endpoints(line, out.shape)
        cv2.line(out, p1, p2, (0, 0, 255), 2)
        overlay_label(out, f"angle={float(slope_deg):+.2f} deg  offset={float(intercept_px):+.1f}px")
        return out

    if isinstance(raw, dict):
        line = raw.get("line")
        if line is not None:
            p1, p2 = line_endpoints(line, out.shape)
            cv2.line(out, p1, p2, (0, 0, 255), 2)

        parts = []
        if "confidence" in raw:
            parts.append(f"conf={float(raw['confidence']):.2f}")
        if "angle_deg" in raw:
            parts.append(f"angle={float(raw['angle_deg']):+.2f} deg")
        if "intercept_y_at_x0" in raw:
            offset = float(raw["intercept_y_at_x0"])
            parts.append("offset=vertical" if np.isnan(offset) else f"offset={offset:+.1f}px")
        overlay_label(out, "  ".join(parts) if parts else "prediction")
        return out

    if isinstance(raw, list):
        palette = [(0, 0, 255), (0, 128, 255), (0, 255, 255), (0, 255, 128), (0, 255, 0)]
        for rank, item in enumerate(raw):
            line = item.get("line")
            if line is None:
                continue
            p1, p2 = line_endpoints(line, out.shape)
            color = palette[rank % len(palette)]
            cv2.line(out, p1, p2, color, max(1, 3 - rank))
            parts = [f"#{rank + 1}"]
            if "confidence" in item:
                parts.append(f"conf={float(item['confidence']):.2f}")
            if "angle_deg" in item:
                parts.append(f"angle={float(item['angle_deg']):+.2f} deg")
            overlay_label(out, "  ".join(parts), y=25 + rank * 22, color=color)
        return out

    overlay_label(out, f"unsupported output type: {type(raw).__name__}", color=(0, 0, 255))
    return out


def annotate_image(module, image_bgr: np.ndarray, raw) -> np.ndarray:
    draw = getattr(module, "draw_horizon", None)
    if draw is None:
        return fallback_draw(image_bgr, raw)

    try:
        if raw is None:
            return fallback_draw(image_bgr, raw)
        if isinstance(raw, tuple) and len(raw) == 3:
            slope_deg, intercept_px, _mask = raw
            return draw(image_bgr, slope_deg, intercept_px)
        if isinstance(raw, dict) and "confidence" in raw:
            return draw(image_bgr, [raw])
        if isinstance(raw, dict) and {"line", "angle_deg", "intercept_y_at_x0"} <= raw.keys():
            return draw(image_bgr, raw["line"], raw["angle_deg"], raw["intercept_y_at_x0"])
        if isinstance(raw, list):
            return draw(image_bgr, raw)
    except Exception:
        return fallback_draw(image_bgr, raw)

    return fallback_draw(image_bgr, raw)


def render_outputs(
    attempt_dir: Path,
    images_dir: Path,
    limit: int | None,
    frame_duration_s: float,
) -> tuple[Path, Path, int]:
    module = load_attempt_module(attempt_dir)
    detect = module.detect_horizon
    image_paths = discover_images(images_dir, limit)

    run_root = attempt_dir / "outputs" / source_name(images_dir)
    frames_dir = run_root / "frames"
    video_path = run_root / "preview.mp4"

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        started = time.perf_counter()
        raw = detect(image)
        elapsed_ms = (time.perf_counter() - started) * 1000

        annotated = annotate_image(module, image, raw)
        overlay_label(annotated, f"{image_path.name}  time={elapsed_ms:.1f}ms", y=annotated.shape[0] - 12)

        output_path = frames_dir / image_path.name
        if not cv2.imwrite(str(output_path), annotated):
            raise RuntimeError(f"Could not write frame: {output_path}")

        if index == 1 or index == total or index % 25 == 0:
            print(f"[{index:>3}/{total}] rendered {output_path.name}")

    frame_count, _, _, _ = stitch_frames_to_video(
        frames_dir,
        video_path,
        frame_duration_s=frame_duration_s,
    )
    return frames_dir, video_path, frame_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path, help="Path to an attempt folder")
    parser.add_argument(
        "--images",
        type=Path,
        default=DEFAULT_IMAGES,
        help="Directory of input images (default: Horizon-UAV images)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only render the first N images")
    parser.add_argument(
        "--frame-duration",
        type=float,
        default=DEFAULT_FRAME_DURATION_S,
        help="Seconds to hold each frame in the preview video (default: 0.5)",
    )
    args = parser.parse_args()

    frames_dir, video_path, frame_count = render_outputs(
        attempt_dir=args.attempt,
        images_dir=args.images,
        limit=args.limit,
        frame_duration_s=args.frame_duration,
    )
    print(f"frames: {frames_dir}")
    print(f"video:  {video_path}")
    print(f"stitched {frame_count} frames")


if __name__ == "__main__":
    main()
