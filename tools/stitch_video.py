"""
tools/stitch_video.py — stitch a directory of frames into a watchable MP4.

Usage:
    .venv/bin/python tools/stitch_video.py outputs/attempt-2-rotation-invariant/samples/frames
    .venv/bin/python tools/stitch_video.py attempts/.../outputs/.../frames --frame-duration 0.75
"""

import argparse
from pathlib import Path

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_FRAME_DURATION_S = 0.5
DEFAULT_CODEC_CANDIDATES = ("avc1", "mp4v")


def list_frames(frames_dir: Path) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frame directory does not exist: {frames_dir}")
    frames = [
        path for path in sorted(frames_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not frames:
        raise FileNotFoundError(f"No image frames found in: {frames_dir}")
    return frames


def open_video_writer(output_path: Path, fps: float, width: int, height: int):
    for codec_name in DEFAULT_CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec_name)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, codec_name
        writer.release()
    raise RuntimeError(
        f"Could not open video writer for {output_path} with any of: "
        f"{', '.join(DEFAULT_CODEC_CANDIDATES)}"
    )


def stitch_frames_to_video(
    frames_dir: Path,
    output_path: Path,
    frame_duration_s: float = DEFAULT_FRAME_DURATION_S,
) -> tuple[int, tuple[int, int], float, str]:
    frames = list_frames(frames_dir)
    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Could not read first frame: {frames[0]}")

    if frame_duration_s <= 0:
        raise ValueError(f"frame_duration_s must be > 0, got {frame_duration_s}")

    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / frame_duration_s

    writer, codec_name = open_video_writer(output_path, fps, width, height)

    try:
        count = 0
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            count += 1
    finally:
        writer.release()

    return count, (width, height), fps, codec_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_dir", type=Path, help="Directory containing image frames")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output MP4 path (default: <frames_dir>.mp4 beside the frames directory)",
    )
    parser.add_argument(
        "--frame-duration",
        type=float,
        default=DEFAULT_FRAME_DURATION_S,
        help="Seconds to hold each frame in the output video (default: 0.5)",
    )
    args = parser.parse_args()

    output_path = args.out or args.frames_dir.parent / f"{args.frames_dir.name}.mp4"
    count, (width, height), fps, codec_name = stitch_frames_to_video(
        args.frames_dir,
        output_path,
        frame_duration_s=args.frame_duration,
    )
    print(
        f"stitched {count} frames into {output_path} "
        f"({width}x{height}, {args.frame_duration:.2f}s/frame, {fps:.2f} fps, codec={codec_name})"
    )


if __name__ == "__main__":
    main()
