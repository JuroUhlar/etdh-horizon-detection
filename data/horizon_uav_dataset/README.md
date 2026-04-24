# Horizon-UAV dataset (mirror)

A 490-image labelled dataset of front-facing UAV footage with per-image sky/land segmentation masks and pre-computed horizon line labels. Used as the primary ground-truth source for evaluating attempts against.

## Origin and attribution

- **Upstream repo:** [NovelioPI/horizon-detection-for-uav](https://github.com/NovelioPI/horizon-detection-for-uav)
- **Upstream dataset home:** https://1drv.ms/f/s!ArMhy7w0Pabls18kV9D7GVtonJmq (OneDrive)
- **Licence:** MIT — © 2021 Novelio Putra Indarto. See the upstream repo's `LICENSE`.
- **What was copied:** the `dataset-slope/` variant only. The three variants in the original download (`dataset`, `dataset-original`, `dataset-slope`) share byte-identical images and masks; only `dataset-slope` additionally ships the pre-computed `label.csv`, so it's a strict superset.

Not mirrored: the three raw `.zip` archives, the LabelBox `export-2020-*.json` (we have the distilled labels + masks), `.DS_Store`.

## Contents

```
data/horizon_uav_dataset/
├── images/                     # 490 JPEGs, all 480×480
│   └── <clip>_<frame>.jpg
├── masks/
│   ├── land/                   # 490 PNGs, 480×480 RGB (use any channel as grayscale)
│   │   └── <clip>_<frame>.png  # white = land, black = sky
│   └── sky/                    # 490 PNGs — bitwise-not of `land/`, redundant but provided
└── label.csv                   # 490 rows, 1 header
```

Filenames are derived from source videos: `<clip>-<timestamp>_<frame>.jpg`. The basename (without extension) is consistent across `images/`, `masks/land/`, `masks/sky/`.

## `label.csv` format

Header: `filename,slope,offset`. Three columns, 490 rows.

The labels describe the horizon as `y = slope·x + c` where `x, y` are raw pixel coordinates:

- **`slope`** — raw `dy/dx` in pixels, as produced by least-squares line fit to mask-boundary pixels. To get roll angle in degrees: `roll_deg = degrees(atan(slope))`. Range in this dataset: `[−1.28, +1.55]`, i.e. roll in roughly `[−52°, +57°]`. **No near-vertical horizons** (no `|roll| > ~60°`), so this dataset does not exercise the rotation-invariance edge case tested by our starter samples 2 and 4.
- **`offset`** — the y-intercept `c` *normalised by image height*. To get pixel y-intercept: `c_px = offset * image_height` (480 in all cases here). A sample with `offset=0.55` has the line crossing x=0 at y≈264 — a slightly-below-centre horizontal horizon.

The convention was reverse-engineered from [`generate_dataset_slope.py`](https://github.com/NovelioPI/horizon-detection-for-uav/blob/master/generate_dataset_slope.py) in the upstream repo:

```python
m, c = np.linalg.lstsq(X, y, rcond=None)[0]   # pixel-space least-squares
c = c / image_height                           # stored offset is normalised
```

## Mask convention

Both `masks/land/*.png` and `masks/sky/*.png` are RGB PNGs, but values are effectively binary: pixels are either `(0,0,0)` or `(255,255,255)`. Read with `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` to get a single-channel uint8 mask.

- `masks/land/<file>.png`: **white = land, black = sky** (the label-polygon target class is "land"; inside the polygon = white).
- `masks/sky/<file>.png`: **white = sky, black = land** (bitwise-not of `land`).

## Known caveats

- **Straight-line approximation.** Labels come from a straight-line least-squares fit to the mask boundary. For images where the boundary curves (distant mountains, buildings), the line is an *approximation* of the true horizon. The masks are the authoritative region-level GT; `label.csv` is the line-level GT consistent with our line-output task.
- **No near-vertical horizons.** The dataset covers roll angles roughly `±57°` only, so it alone cannot validate rotation-invariance claims. Our `data/samples/` serve as the near-vertical stress test.
- **Mask non-binary edges?** Worth checking if any PNGs have anti-aliased grey values. If yes, threshold at 127 when loading. (Spot-checked one sample and it's cleanly binary, but do not assume across all 490.)

## Intended use in this repo

- **Numerical evaluation** of attempt N's `detect_horizon` against `label.csv`, reporting errors in Hesse form (`(θ, ρ)`) as described in [`docs/evaluation-metrics.md`](../../docs/evaluation-metrics.md).
- **IoU evaluation** of predicted sky masks against `masks/sky/` (or `masks/land/`).
- **Parameter tuning** for classical pipelines (train/test split can be per-clip to avoid leakage across sequential frames).
