"""
tools/annotate_horizon.py — interactive horizon-line annotator.

Click two points on the horizon in each image; the tool fits the line
y = slope*x + c in original-image pixel space and writes the result to
label.csv. Schema (a superset of data/horizon_uav_dataset/label.csv,
which is missing the has_horizon column — old 3-column CSVs are read
back as has_horizon=true and rewritten in the 4-column form on save):

    filename,has_horizon,slope,offset
    foo.jpg,true,<dy/dx in px>,<c / image_height>
    bar.jpg,false,,                              # all-sky or all-ground frame

Usage:
    .venv/bin/python tools/annotate_horizon.py
    .venv/bin/python tools/annotate_horizon.py --dataset data/video_clips_ukraine_atv
    .venv/bin/python tools/annotate_horizon.py --relabel       # revisit already-labeled images
    .venv/bin/python tools/annotate_horizon.py --start 42      # jump to index 42

Keys:
    left-click x2  define horizon (first point, second point)
    n / Enter      save current label and advance
    x              mark image as NO HORIZON (sky-only or ground-only) and advance
    u              undo last click
    r              reset both points
    b              go back to previous image (does not erase its label)
    s              skip (advance without saving a label for this image)
    q / Esc        save state and quit
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "video_clips_ukraine_atv"

WINDOW_NAME = "horizon annotator"
MAX_DISPLAY_W = 1600
MAX_DISPLAY_H = 900


def load_existing_labels(csv_path: Path) -> dict[str, dict]:
    """Read label.csv into {filename: {"has_horizon", "slope", "offset"}}.

    Backward-compatible with the 3-column schema (filename,slope,offset):
    rows lacking a has_horizon column are loaded as has_horizon=true.
    """
    if not csv_path.exists():
        return {}
    out: dict[str, dict] = {}
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        has_col = "has_horizon" in (reader.fieldnames or [])
        for row in reader:
            hh = (row["has_horizon"].strip().lower() == "true") if has_col else True
            if hh:
                out[row["filename"]] = {
                    "has_horizon": True,
                    "slope": float(row["slope"]),
                    "offset": float(row["offset"]),
                }
            else:
                out[row["filename"]] = {"has_horizon": False, "slope": None, "offset": None}
    return out


def write_labels(csv_path: Path, labels: dict[str, dict]) -> None:
    # Stable order: by filename, matching how the reference dataset is laid out.
    rows = sorted(labels.items(), key=lambda kv: kv[0])
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "has_horizon", "slope", "offset"])
        for name, lbl in rows:
            if lbl["has_horizon"]:
                writer.writerow([name, "true", repr(lbl["slope"]), repr(lbl["offset"])])
            else:
                writer.writerow([name, "false", "", ""])
    tmp.replace(csv_path)  # atomic-ish; protects against crashes mid-write.


def fit_to_window(img: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale < 1.0:
        disp = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        disp = img.copy()
    return disp, scale


def line_endpoints_at_image_edges(
    p1: tuple[float, float], p2: tuple[float, float], w: int, h: int
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Extend the line through p1, p2 to the image borders for drawing."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return None
    if abs(dx) >= abs(dy):
        # Mostly horizontal: solve at x=0 and x=w-1.
        m = dy / dx
        y_at_0 = y1 - m * x1
        y_at_w = y1 + m * (w - 1 - x1)
        return (0, int(round(y_at_0))), (w - 1, int(round(y_at_w)))
    # Mostly vertical: solve at y=0 and y=h-1.
    m = dx / dy
    x_at_0 = x1 - m * y1
    x_at_h = x1 + m * (h - 1 - y1)
    return (int(round(x_at_0)), 0), (int(round(x_at_h)), h - 1)


def render_overlay(
    base: np.ndarray,
    points_disp: list[tuple[int, int]],
    hover_disp: tuple[int, int] | None,
    header_text: str,
    sub_text: str,
) -> np.ndarray:
    canvas = base.copy()
    h, w = canvas.shape[:2]

    # Live preview: if we have one click + a mouse hover, treat hover as the second point.
    preview_pts: list[tuple[int, int]] = list(points_disp)
    if len(preview_pts) == 1 and hover_disp is not None:
        preview_pts.append(hover_disp)

    if len(preview_pts) == 2:
        ends = line_endpoints_at_image_edges(preview_pts[0], preview_pts[1], w, h)
        if ends is not None:
            color = (0, 255, 255) if len(points_disp) == 2 else (0, 200, 200)
            cv2.line(canvas, ends[0], ends[1], color, 2, cv2.LINE_AA)

    for i, (px, py) in enumerate(points_disp):
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, str(i + 1), (px + 8, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

    # Header bar with progress + filename, second line with current slope/offset or hint.
    cv2.rectangle(canvas, (0, 0), (w, 56), (0, 0, 0), -1)
    cv2.putText(
        canvas, header_text, (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, sub_text, (10, 46),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA,
    )
    return canvas


def compute_slope_offset(
    p1: tuple[float, float], p2: tuple[float, float], image_height: int
) -> tuple[float, float] | None:
    """Returns (slope, offset) in the same convention as horizon_uav_dataset/label.csv.

    slope  = dy/dx in raw pixels.
    offset = y-intercept c (in pixels), normalised by image height.
    """
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:
        # Vertical horizon — infinite slope; the reference format can't represent it.
        return None
    slope = (y2 - y1) / (x2 - x1)
    c = y1 - slope * x1
    offset = c / image_height
    return slope, offset


def annotate(dataset_dir: Path, relabel: bool, start_index: int) -> None:
    images_dir = dataset_dir / "images"
    csv_path = dataset_dir / "label.csv"

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    labels = load_existing_labels(csv_path)
    print(f"Loaded {len(labels)} existing labels from {csv_path}")
    print(f"Found {len(image_paths)} images in {images_dir}")

    # Build the working queue.
    if relabel:
        queue_indices = list(range(start_index, len(image_paths)))
    else:
        queue_indices = [
            i for i in range(start_index, len(image_paths))
            if image_paths[i].name not in labels
        ]
    if not queue_indices:
        print("Nothing to label (use --relabel to revisit, or --start to jump back).")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Mouse state lives in a dict so the callback can mutate it.
    state = {
        "points_disp": [],   # clicks in display coordinates
        "points_full": [],   # same clicks mapped to full-resolution coordinates
        "hover_disp": None,
        "scale": 1.0,
        "needs_redraw": True,
    }

    def on_mouse(event, x, y, flags, _userdata):
        if event == cv2.EVENT_MOUSEMOVE:
            if state["hover_disp"] != (x, y):
                state["hover_disp"] = (x, y)
                if len(state["points_disp"]) == 1:
                    state["needs_redraw"] = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            if len(state["points_disp"]) >= 2:
                return  # already have both endpoints; user must press u/r/n
            state["points_disp"].append((x, y))
            inv = 1.0 / state["scale"]
            state["points_full"].append((x * inv, y * inv))
            state["needs_redraw"] = True

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    # Position within queue_indices, not within image_paths — that way Back stays inside the work queue.
    cursor = 0

    while 0 <= cursor < len(queue_indices):
        idx = queue_indices[cursor]
        path = image_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            print(f"  ! could not read {path.name}, skipping")
            cursor += 1
            continue

        h_full, w_full = img.shape[:2]
        disp_base, scale = fit_to_window(img)
        state["scale"] = scale
        state["points_disp"] = []
        state["points_full"] = []
        state["hover_disp"] = None
        state["needs_redraw"] = True

        # If a label already exists (relabel mode), seed it so the user sees the prior line.
        prior = labels.get(path.name)

        while True:
            if state["needs_redraw"] or prior is not None:
                # Compute current slope/offset preview.
                pts_full = list(state["points_full"])
                preview_pts_full = pts_full
                if len(pts_full) == 1 and state["hover_disp"] is not None:
                    hx, hy = state["hover_disp"]
                    preview_pts_full = pts_full + [(hx / scale, hy / scale)]

                if len(preview_pts_full) == 2:
                    res = compute_slope_offset(preview_pts_full[0], preview_pts_full[1], h_full)
                    sub = (
                        f"slope={res[0]:+.4f}  offset={res[1]:+.4f}"
                        if res is not None
                        else "vertical line — not representable, click different x"
                    )
                elif prior is not None and not state["points_disp"]:
                    if prior["has_horizon"]:
                        sub = f"prior: slope={prior['slope']:+.4f}  offset={prior['offset']:+.4f}  (click to overwrite)"
                    else:
                        sub = "prior: NO HORIZON  (click to overwrite, or x to keep)"
                else:
                    sub = "click 2 pts  |  n=save  x=no-horizon  u=undo  r=reset  b=back  s=skip  q=quit"

                header = f"[{cursor + 1}/{len(queue_indices)}]  idx={idx}  {path.name}  ({w_full}x{h_full})"

                # If we have a prior label and no new clicks yet, draw the prior line as a hint.
                draw_pts = list(state["points_disp"])
                canvas = render_overlay(disp_base, draw_pts, state["hover_disp"], header, sub)
                if prior is not None and not state["points_disp"]:
                    if prior["has_horizon"]:
                        slope_p, offset_p = prior["slope"], prior["offset"]
                        c_px = offset_p * h_full
                        p1_full = (0.0, c_px)
                        p2_full = (float(w_full - 1), slope_p * (w_full - 1) + c_px)
                        p1_disp = (int(round(p1_full[0] * scale)), int(round(p1_full[1] * scale)))
                        p2_disp = (int(round(p2_full[0] * scale)), int(round(p2_full[1] * scale)))
                        ends = line_endpoints_at_image_edges(p1_disp, p2_disp, canvas.shape[1], canvas.shape[0])
                        if ends is not None:
                            cv2.line(canvas, ends[0], ends[1], (255, 0, 255), 1, cv2.LINE_AA)
                    else:
                        # Big banner so you can't miss that this image is currently labelled no-horizon.
                        ch, cw = canvas.shape[:2]
                        cv2.putText(
                            canvas, "NO HORIZON", (cw // 2 - 180, ch // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 255), 4, cv2.LINE_AA,
                        )

                cv2.imshow(WINDOW_NAME, canvas)
                state["needs_redraw"] = False
                prior_drawn_once = True  # only the very first redraw needs to render the prior line

            key = cv2.waitKey(20) & 0xFF

            # No key pressed; loop again to pick up mouse moves.
            if key == 255:
                continue

            if key in (ord("q"), 27):  # q or Esc
                cv2.destroyAllWindows()
                write_labels(csv_path, labels)
                print(f"Saved {len(labels)} labels to {csv_path}. Bye.")
                return

            if key == ord("u"):
                if state["points_disp"]:
                    state["points_disp"].pop()
                    state["points_full"].pop()
                    state["needs_redraw"] = True
                continue

            if key == ord("r"):
                state["points_disp"] = []
                state["points_full"] = []
                state["needs_redraw"] = True
                continue

            if key == ord("b"):
                if cursor > 0:
                    cursor -= 1
                break  # leave inner loop, reload prior image

            if key == ord("s"):
                cursor += 1
                break

            if key == ord("x"):
                labels[path.name] = {"has_horizon": False, "slope": None, "offset": None}
                write_labels(csv_path, labels)
                print(f"  [{cursor + 1}/{len(queue_indices)}] {path.name}  NO HORIZON")
                cursor += 1
                break

            if key in (ord("n"), 13, 10):  # n, Enter (LF or CR)
                if len(state["points_full"]) != 2:
                    print("  need two clicks before saving (or press x for no-horizon)")
                    continue
                res = compute_slope_offset(state["points_full"][0], state["points_full"][1], h_full)
                if res is None:
                    print("  vertical line — pick two points with different x")
                    continue
                slope, offset = res
                labels[path.name] = {"has_horizon": True, "slope": slope, "offset": offset}
                write_labels(csv_path, labels)
                print(f"  [{cursor + 1}/{len(queue_indices)}] {path.name}  slope={slope:+.4f}  offset={offset:+.4f}")
                cursor += 1
                break

    cv2.destroyAllWindows()
    write_labels(csv_path, labels)
    print(f"Done. Saved {len(labels)} labels to {csv_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="dataset directory containing images/ (default: data/video_clips_ukraine_atv)")
    parser.add_argument("--relabel", action="store_true",
                        help="iterate over all images, including ones already in label.csv")
    parser.add_argument("--start", type=int, default=0,
                        help="start at this index in the sorted image list")
    args = parser.parse_args()
    annotate(args.dataset, args.relabel, args.start)


if __name__ == "__main__":
    main()
