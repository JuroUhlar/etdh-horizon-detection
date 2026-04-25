# Attempt 1 — Brightness Threshold + Simple Line Fit

This is the baseline. It asks a very simple question:

"Can we split the image into bright sky and darker ground, then draw one straight line through that boundary?"

It is extremely fast and easy to understand, but it breaks when that simple assumption is wrong.

## Technique in Plain English

1. Convert the image to grayscale.
2. Blur it slightly so small texture does not confuse the split.
3. Use Otsu thresholding to divide the frame into "bright" and "dark" regions.
4. Clean the mask with morphology so tiny holes do not dominate the result.
5. For each image column, find where sky stops and ground starts.
6. Fit one straight line through those boundary points.

That means this attempt depends on two big assumptions:

- Sky is brighter than ground.
- Sky is above ground in the image.

When either assumption fails, the fitted line can be very wrong.

## What The Metric Labels Mean

These labels also appear in the evaluator output:

- **Pass rate**: the share of images where the prediction is good enough by both angle and position. In this repo that means `Δθ < 5°` and `Δρ / H < 5%`.
- **Δθ**: angle error. How many degrees the predicted horizon is tilted away from the ground-truth horizon.
- **Δρ**: position error in pixels. How far the predicted line is shifted from the ground-truth line.
- **Δρ / H**: the same position error, but divided by image height so different image sizes are comparable.
- **Sky-mask IoU**: overlap between predicted sky region and ground-truth sky region. `1.0` is perfect.
- **Mean latency**: average runtime per image.

Smaller is better for `Δθ`, `Δρ`, `Δρ / H`, and latency. Bigger is better for pass rate and IoU.

## Full-Dataset Results

Measured on the Horizon-UAV dataset (`490` images) with:

```bash
.venv/bin/python tools/evaluate.py attempts/attempt-1-otsu-column-scan
```

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Δθ (angle error, deg) | 10.461 | 1.415 | 36.793 | 85.059 |
| Δρ (line position error, px) | 70.744 | 10.990 | 274.909 | 538.325 |
| Δρ / H (normalised line position error) | 0.147 | 0.023 | 0.573 | 1.122 |
| Sky-mask IoU | 0.926 | 0.950 | 0.984 | 0.997 |
| Latency (ms) | 0.757 | 0.674 | 0.743 | 20.664 |

**Pass rate:** `306 / 490 = 62.4%`

## What These Numbers Mean

- The median frame is actually decent. P50 angle error is only `1.415°`.
- The bad news is the tail. P90 angle error is `36.793°`, which means a meaningful chunk of frames are way off.
- The sky mask overlap is already high (`0.926` mean IoU), so the segmentation is often roughly right even when the fitted line is bad.
- It is extremely fast: `0.757 ms` mean latency on the dev machine.

In plain terms: this attempt is a good "cheap baseline", but it is not reliable enough.

## Main Failure Modes

- **Rotated scenes**: the column scan assumes sky is above ground, so it breaks badly when the frame is rotated.
- **Low-contrast scenes**: if sky and ground have similar brightness, Otsu can split the image along the wrong boundary.
- **Outliers**: one bad region can pull the fitted line away because the fit is not robust.

## Bottom Line

Attempt 1 is useful because it proves the basic pipeline can work, and it shows what "very fast but brittle" looks like. It is the speed champion, but not the accuracy champion.
