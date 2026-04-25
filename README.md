# ETDH Horizon Detection

Lightweight horizon detection experiments for a UAV hackathon-style task: find the horizon line, or a sky mask, cheaply enough to run on embedded hardware and remove sky pixels before a downstream detector runs.

This repo is not a packaged library. It is an evaluation sandbox with:

- three classical computer-vision attempts under `attempts/`
- a dataset mirror under `data/horizon_uav_dataset/`
- a small rotated stress set under `data/samples/`
- a Ukraine ATV FPV clip set under `data/video_clips_ukraine_atv/` (videos + extracted frames, hand-labelled with `tools/annotate_horizon.py`)
- an evaluator in `tools/evaluate.py`
- visual rendering utilities in `tools/render_outputs.py` and `tools/stitch_video.py`
- an interactive annotator in `tools/annotate_horizon.py` for labelling new frame sets in the same `label.csv` schema
- notes in `docs/` plus a plain-language comparison in `attempt_comparison.md`

## Current Status

The three attempts tell a clear story:

- `attempt-1-otsu-column-scan`: fastest, but brittle
- `attempt-2-rotation-invariant`: much better line fitting, still mask-limited
- `attempt-3-top-n-ransac`: best accuracy by far, much slower

On the 490-image Horizon-UAV dataset:

| Attempt | Pass rate | Mean angle error | Mean position error | Mean latency |
|---|---:|---:|---:|---:|
| Attempt 1 | 62.4% | 10.461 deg | 70.744 px | 0.762 ms |
| Attempt 2 | 81.2% | 7.313 deg | 36.700 px | 3.689 ms |
| Attempt 3 | 95.9% | 1.067 deg | 10.102 px | 69.553 ms |

The full breakdown lives in [attempt_comparison.md](./attempt_comparison.md).

## Quick Start

```bash
python -m venv .venv
.venv/bin/pip install numpy opencv-python
```

Run one attempt on a sample image:

```bash
.venv/bin/python attempts/attempt-2-rotation-invariant/horizon_detect.py data/samples/sample1.jpg
```

Evaluate an attempt on the full dataset:

```bash
.venv/bin/python tools/evaluate.py attempts/attempt-3-top-n-ransac
```

Use `--limit` for quick iteration:

```bash
.venv/bin/python tools/evaluate.py attempts/attempt-3-top-n-ransac --limit 50
```

Render annotated frames and a preview video:

```bash
.venv/bin/python tools/render_outputs.py attempts/attempt-2-rotation-invariant --images data/samples
```

That writes:

- `attempts/<attempt>/outputs/<source>/frames/*.jpg`
- `attempts/<attempt>/outputs/<source>/preview.mp4`

You can also restitch any frame directory manually:

```bash
.venv/bin/python tools/stitch_video.py attempts/attempt-2-rotation-invariant/outputs/samples/frames --frame-duration 0.75
```

The default preview pacing is `0.5s` per frame (`2 fps`). Override it with `--frame-duration <seconds>`.

## Annotating a New Dataset

`tools/annotate_horizon.py` is a small OpenCV GUI for hand-labelling horizon lines on a folder of images. It writes `label.csv` in the same `filename,slope,offset` schema as `data/horizon_uav_dataset/label.csv`, so labels produced this way drop straight into `tools/evaluate.py` without any conversion step.

Run it against the default dataset (`data/video_clips_ukraine_atv/`):

```bash
.venv/bin/python tools/annotate_horizon.py
```

Or point it at any directory containing an `images/` subfolder:

```bash
.venv/bin/python tools/annotate_horizon.py --dataset data/my_new_dataset
```

Workflow per image:

1. Left-click two points along the horizon. The line is extended to the image borders so you can sanity-check the geometry before saving.
2. Press `n` (or Enter) to save the label and advance.
3. If the frame contains only sky or only ground (no visible horizon), press `x` instead — the row is written with `has_horizon=false` and empty slope/offset.
4. The header bar shows the current `slope` and `offset` so you see exactly what will be written.

Keys:

| Key | Action |
|---|---|
| `n` / Enter | save current label and advance |
| `x` | mark image as NO HORIZON (sky-only or ground-only) and advance |
| `u` | undo last click |
| `r` | reset both points |
| `b` | go back to previous image (does not erase its label) |
| `s` | skip without labelling |
| `q` / Esc | save state and quit |

Other flags:

- `--relabel` — iterate over all images, including ones already in `label.csv` (the prior line is shown in magenta so you can decide whether to overwrite or `s`-skip).
- `--start <i>` — jump to index `i` in the sorted image list, useful for resuming partway in.

By default, already-labelled images are skipped, and `label.csv` is rewritten atomically after every save, so you can stop and resume freely without losing progress.

The on-disk convention is a superset of the upstream Horizon-UAV labels:

```
filename,has_horizon,slope,offset
foo.jpg,true,-0.128,0.323
bar.jpg,false,,                 # sky-only / ground-only frame
```

For horizon rows, `slope = dy/dx` in raw pixels and `offset = y_intercept_px / image_height`, exactly as in the upstream dataset — see [data/horizon_uav_dataset/README.md](./data/horizon_uav_dataset/README.md#labelcsv-format) for the full derivation. The upstream `data/horizon_uav_dataset/label.csv` is byte-identical to upstream and lacks the `has_horizon` column; loaders treat its rows as `has_horizon=true`.

## Repository Layout

```text
attempts/
  attempt-1-otsu-column-scan/
  attempt-2-rotation-invariant/
  attempt-3-top-n-ransac/
data/
  horizon_uav_dataset/         # 490 labelled images + masks + label.csv
  samples/                     # 4 manual stress-test images, especially rotation edge cases
  video_clips_ukraine_atv/     # FPV clips + 121 extracted frames; label.csv built with annotate_horizon.py
docs/
  evaluation-metrics.md
  research-horizon-detection.md
  inspiration-implementations.md
tools/
  evaluate.py
  render_outputs.py
  stitch_video.py
  annotate_horizon.py
```

## Detector Contract

`tools/evaluate.py` dynamically imports `horizon_detect.py` from an attempt directory and calls:

```python
detect_horizon(image_bgr: np.ndarray)
```

Accepted return shapes are:

- `None` — detector failed / gave up (counted distinctly from a deliberate "no horizon")
- `"no_horizon"` (string) — detector deliberately reports the frame has no horizon (sky-only or ground-only)
- `{"no_horizon": True, "mask": mask}` — same, in dict form, with optional sky mask
- `(slope_deg, intercept_px, mask)` for the simple baseline style
- `{"line": (vx, vy, x0, y0), "mask": mask, ...}` for rotation-safe line output

This loose contract is intentional: each attempt stays self-contained instead of becoming a package. Note that none of the three current attempts emit no-horizon decisions yet — they always predict a line, which means they take a confusion-matrix hit on no-horizon labels in the Ukraine ATV dataset.

## Metrics

The evaluator scores line accuracy in Hesse normal form, not raw slope/intercept, so near-vertical lines do not blow up numerically. It reports:

- angular error `Δθ`
- positional error `Δρ`
- normalised positional error `Δρ / H`
- sky-mask IoU when a mask is returned
- latency
- pass rate with `Δθ < 5°` and `Δρ / H < 5%`
- when the dataset contains no-horizon labels: a 2×2 confusion matrix (TP/FN/FP/TN over `has_horizon`), and pass-rate folds in classification agreement (a frame "passes" only if `gt_has_horizon == pred_has_horizon` AND, when both are horizons, line errors are within thresholds)

Details are in [docs/evaluation-metrics.md](./docs/evaluation-metrics.md).

## Data Notes

- `data/horizon_uav_dataset/` is the main benchmark and includes images, land/sky masks, and `label.csv`.
- `data/samples/` is a tiny manual stress set for rotated or near-vertical cases that the main dataset does not cover well.
- `data/video_clips_ukraine_atv/` holds short FPV clips (`clips/`) plus 121 extracted frames (`images/`); labels in `label.csv` are produced by `tools/annotate_horizon.py` and follow the same schema as the main benchmark.
- The dataset mirror has its own README at [data/horizon_uav_dataset/README.md](./data/horizon_uav_dataset/README.md).

## If You Add Another Attempt

Put it under `attempts/<name>/horizon_detect.py` and keep it directly runnable from the CLI. If you want it to work with the evaluator, expose `detect_horizon(image_bgr)` using one of the accepted return shapes above.
