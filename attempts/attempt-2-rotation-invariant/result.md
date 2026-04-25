# Attempt 2 — Rotation-Invariant Boundary + Robust Line Fit

This attempt keeps the same sky-vs-ground mask as attempt 1, but fixes the line-fitting stage.

The core idea is:

"Instead of scanning from the top of the image downward, find the whole boundary shape directly, then fit a line to it in a way that is less sensitive to weird points."

That makes it much better on rotated images and somewhat better on noisy boundaries.

## Technique in Plain English

1. Convert the image to grayscale.
2. Blur it slightly.
3. Use Otsu thresholding to split likely sky from likely ground.
4. Clean the mask with close/open morphology.
5. Extract the sky-ground boundary using a morphological gradient.
6. Fit a line through those boundary pixels with OpenCV's `fitLine` using Huber loss.

Compared with attempt 1:

- The boundary finder is now orientation-agnostic.
- The fitted line is more robust to outliers.
- The sky/ground classifier itself is still the same.

So this attempt mainly fixes "where we draw the line", not "how we classify sky".

## What The Metric Labels Mean

- **Pass rate**: the share of images where the prediction is close enough in both angle and position. Here that means `Δθ < 5°` and `Δρ / H < 5%`.
- **Δθ**: angle error in degrees.
- **Δρ**: line position error in pixels.
- **Δρ / H**: line position error divided by image height.
- **Sky-mask IoU**: overlap between predicted sky mask and the ground-truth sky mask.
- **Mean latency**: average runtime per image.

Smaller is better for `Δθ`, `Δρ`, `Δρ / H`, and latency. Bigger is better for pass rate and IoU.

## Full-Dataset Results

Measured on the Horizon-UAV dataset (`490` images) with:

```bash
.venv/bin/python tools/evaluate.py attempts/attempt-2-rotation-invariant
```

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Δθ (angle error, deg) | 7.313 | 0.917 | 32.036 | 88.744 |
| Δρ (line position error, px) | 36.700 | 7.767 | 101.693 | 616.916 |
| Δρ / H (normalised line position error) | 0.076 | 0.016 | 0.212 | 1.285 |
| Sky-mask IoU | 0.929 | 0.952 | 0.984 | 0.997 |
| Latency (ms) | 3.689 | 2.628 | 6.907 | 28.770 |

**Pass rate:** `398 / 490 = 81.2%`

## Improvement Over Attempt 1

| Metric | Attempt 1 | Attempt 2 | Change |
|---|---:|---:|---:|
| Pass rate | 62.4% | 81.2% | +18.8 pts |
| Mean Δθ | 10.461° | 7.313° | better |
| Mean Δρ | 70.744 px | 36.700 px | much better |
| Mean latency | 0.762 ms | 3.689 ms | slower, but still fast |

## What These Numbers Mean

- This is a clear upgrade over attempt 1.
- The pass rate jumps from `62.4%` to `81.2%`.
- The average angle error and line-position error both improve a lot.
- The sky-mask IoU barely changes, which makes sense because the mask-generation step is still almost the same.
- Runtime is still low in absolute terms, but it is no longer "basically free".

In plain terms: attempt 2 is much better at drawing the line once it has a reasonable sky mask, but it still struggles when the brightness-based mask is fundamentally wrong.

## Main Failure Modes

- **Bad mask, good fitter**: if the brightness split is wrong, this attempt can still fit a clean line to the wrong boundary.
- **Harsh lighting or haze**: bright glare and low-contrast sky/ground still confuse the mask.
- **Worst-case outliers still exist**: Huber helps, but it is not a full candidate search.

## Bottom Line

Attempt 2 is the best speed/quality tradeoff among the first two versions. It fixes the rotation problem and improves robustness without adding much complexity, but it does not yet solve the hardest scenes.

## Train/Test Evaluation

seed=42 | train=388 | test=102

| Metric | Train | Test |
|---|---|---|
| N evaluated | 388 | 102 |
| N failed | 0 | 0 |
| FPS (excl. 10-frame warmup) | 315.2 | 304.9 |
| Mean latency (ms) | 3.17 | 3.28 |
| Mean angle error (°) | 7.71 | 5.80 |
| P90 angle error (°) | 32.88 | 23.37 |
| Mean position error (%H) | 7.95 | 6.48 |
| P90 position error (%H) | 21.23 | 11.21 |
| Mean IoU | 0.928 | 0.931 |
| Pass rate (Δθ<5° & Δρ<5%H) | 80.7% | 83.3% |
| mAP (threshold sweep) | 0.6920 | 0.7255 |

**mAP threshold breakdown:**

| Δθ max | Δρ/H max | Train precision | Test precision |
|---|---|---|---|
| 1° | 1% | 0.131 | 0.167 |
| 2° | 2% | 0.528 | 0.559 |
| 3° | 3% | 0.742 | 0.775 |
| 5° | 5% | 0.807 | 0.833 |
| 7° | 7% | 0.814 | 0.853 |
| 10° | 10% | 0.830 | 0.873 |
| 15° | 15% | 0.838 | 0.873 |
| 20° | 20% | 0.845 | 0.873 |

