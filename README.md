# ETDH Horizon Detection

Lightweight horizon detection experiments for a UAV hackathon-style task: find the horizon line, or a sky mask, cheaply enough to run on embedded hardware and remove sky pixels before a downstream detector runs.

This repo is not a packaged library. It is an evaluation sandbox with:

- three classical computer-vision attempts under `attempts/`
- a dataset mirror under `data/horizon_uav_dataset/`
- a small rotated stress set under `data/samples/`
- an evaluator in `tools/evaluate.py`
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

## Repository Layout

```text
attempts/
  attempt-1-otsu-column-scan/
  attempt-2-rotation-invariant/
  attempt-3-top-n-ransac/
data/
  horizon_uav_dataset/   # 490 labelled images + masks + label.csv
  samples/               # 4 manual stress-test images, especially rotation edge cases
docs/
  evaluation-metrics.md
  research-horizon-detection.md
  inspiration-implementations.md
tools/
  evaluate.py
```

## Detector Contract

`tools/evaluate.py` dynamically imports `horizon_detect.py` from an attempt directory and calls:

```python
detect_horizon(image_bgr: np.ndarray)
```

Accepted return shapes are:

- `None`
- `(slope_deg, intercept_px, mask)` for the simple baseline style
- `{"line": (vx, vy, x0, y0), "mask": mask, ...}` for rotation-safe line output

This loose contract is intentional: each attempt stays self-contained instead of becoming a package.

## Metrics

The evaluator scores line accuracy in Hesse normal form, not raw slope/intercept, so near-vertical lines do not blow up numerically. It reports:

- angular error `Δθ`
- positional error `Δρ`
- normalised positional error `Δρ / H`
- sky-mask IoU when a mask is returned
- latency
- pass rate with `Δθ < 5°` and `Δρ / H < 5%`

Details are in [docs/evaluation-metrics.md](./docs/evaluation-metrics.md).

## Data Notes

- `data/horizon_uav_dataset/` is the main benchmark and includes images, land/sky masks, and `label.csv`.
- `data/samples/` is a tiny manual stress set for rotated or near-vertical cases that the main dataset does not cover well.
- The dataset mirror has its own README at [data/horizon_uav_dataset/README.md](./data/horizon_uav_dataset/README.md).

## If You Add Another Attempt

Put it under `attempts/<name>/horizon_detect.py` and keep it directly runnable from the CLI. If you want it to work with the evaluator, expose `detect_horizon(image_bgr)` using one of the accepted return shapes above.
