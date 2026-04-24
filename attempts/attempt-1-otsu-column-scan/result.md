# Attempt 1 — Otsu threshold + morphological closing + column-scan line fit

Baseline classical-CV pipeline. Purpose: establish a fast, trivial-to-reason-about reference that every later attempt can be benchmarked against.

## Method

1. **Grayscale** — sky/ground differ mainly in brightness.
2. **Gaussian blur (5×5)** — suppress fine texture so thresholding latches onto the global split.
3. **Otsu threshold** — picks a cutoff automatically by maximising between-class variance of the brightness histogram. Sky → 255, ground → 0.
4. **Morphological closing (9×9 rect)** — fill small dark holes inside the sky mask.
5. **Column scan** — for each column, find the first row where sky ends (`np.argmax` on the inverted mask). Skip columns that are all-sky or all-ground.
6. **Least-squares line fit** (`np.polyfit(..., 1)`) across all valid (x, y) boundary samples. Convert slope to degrees.

Output: `(slope_deg, intercept_px)` and an annotated image with the fitted line drawn.

## Parameters

| Param | Value | Why |
|---|---|---|
| Blur kernel | 5×5 Gaussian | Kills leaf/grass texture without smearing the horizon |
| Threshold | Otsu (auto) | Bimodal sky/ground histograms |
| Morphology kernel | 9×9 rect, close | Fills dark specks; small enough not to merge distant regions |
| Line fit | `np.polyfit` deg=1 | Plain least-squares; no outlier rejection |

## How to reproduce

From the repo root:

```bash
# one-off
.venv/bin/python attempts/attempt-1-otsu-column-scan/horizon_detect.py \
    data/samples/sample1.jpg \
    --out attempts/attempt-1-otsu-column-scan/outputs/sample1_horizon.jpg

# batch
for i in 1 2 3 4; do
  .venv/bin/python attempts/attempt-1-otsu-column-scan/horizon_detect.py \
    "data/samples/sample${i}.jpg" \
    --out "attempts/attempt-1-otsu-column-scan/outputs/sample${i}_horizon.jpg"
done
```

## Results

| Sample | Angle (deg) | Offset (px) | Time (ms) | Verdict |
|---|---:|---:|---:|---|
| sample1.jpg | −20.43 | 342.5 | 9.0 | Badly biased — line passes through the frame but at the wrong angle and offset |
| sample2.jpg | +45.44 | −58.6 | 1.9 | Noisy fit — image is ~90° rotated, assumption violated |
| sample3.jpg |   0.00 |    0.1 | 2.0 | Degenerate — image is 180° rotated, line pinned to y=0 |
| sample4.jpg |   0.00 |   −0.0 | 1.9 | Degenerate — image is ~90° rotated, line pinned to y=0 |

No ground-truth annotations available yet, so errors are assessed visually from the annotated outputs in [`outputs/`](outputs/). Timings are on an M-series Mac, not on the Raspberry Pi 5 target.

## Full-dataset scoreboard (Horizon-UAV, 490 images)

Numerical evaluation against the mask-derived `label.csv` from the Horizon-UAV dataset, using `tools/evaluate.py`. Metrics are defined in [`docs/evaluation-metrics.md`](../../docs/evaluation-metrics.md) (`Δθ` = mod-180 angular error in degrees; `Δρ` = positional error in canonical Hesse form, in pixels; `Δρ/H` = same, normalised by image height; IoU = intersection-over-union of the predicted sky mask against the GT sky mask).

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Δθ (deg) | 10.46 | 1.41 | 36.79 | 85.06 |
| Δρ (px) | 70.74 | 10.99 | 274.91 | 538.32 |
| Δρ / H | 14.7% | 2.3% | 57.3% | 112.2% |
| Sky-mask IoU | 0.926 | 0.950 | 0.984 | 0.997 |
| Latency (ms) | 0.77 | 0.68 | 0.79 | 38.5 |

**Pass rate (Δθ < 5° AND Δρ/H < 5%): 306/490 = 62.4%.**

Reproduce with:

```bash
.venv/bin/python tools/evaluate.py attempts/attempt-1-otsu-column-scan
```

### What these numbers tell us

- **The median sample is handled correctly** (P50 Δθ = 1.4°, P50 Δρ/H = 2.3%) — Otsu + column-scan *works* on typical UAV footage where sky-on-top and "sky is brighter" both hold.
- **The tail is catastrophic.** P90 Δθ = 37°, worst 85° — on 10% of frames the fit is off by tens of degrees, and ~38% of frames fail the pass gate.
- **Sky-mask IoU is already 0.93 mean / 0.95 median** — the *classifier* is mostly correct even when the *fitter* produces garbage. This isolates where the errors are coming from: the line-extraction step, not the segmentation.

### Worst-case analysis

Three representative worst offenders (annotated predictions saved in [`outputs/worst_cases/`](outputs/worst_cases/)):

| Image | Δθ | IoU | Root cause |
|---|---:|---:|---|
| `a4-…_0.jpg` | 85.06° | 0.877 | High-roll overcast scene: sky (dull grey) and ground (dark greens) have near-identical luminance, so Otsu splits the image along some other feature entirely. IoU stays high because the right number of pixels end up on each side — just along the wrong dividing line. |
| `e6-…_140.jpg` | 75.21° | 0.613 | Bright sun glare on the right half of the sky. Otsu isolates the glare *as "sky"* and lumps real sky (clouds) in with the ground. Fit tracks the glare boundary, not the horizon. |
| `d6-…_140.jpg` | 75.05° | 0.613 | Near-duplicate of the e6 scene; same failure mode. |

**Pattern:** every worst case is a *classifier* failure, not a *fitter* failure. Brightness is an unreliable sky/ground cue under overcast sky (sky≈ground luminance) or direct sun (bright sub-region inside the sky). The fitter then dutifully fits a line to the wrong boundary. This is the primary target for attempt 3.

## Failure modes (the interesting part)

The baseline encodes three assumptions. The four samples violate all of them:

1. **"Sky is brighter than ground."** Usually true, but **sample 1** has a washed-out haze band just above the real horizon. Otsu's single brightness cutoff appears to split the image as *"bright upper sky"* vs *"hazy lower sky + ground"* — so the column-scan finds the bottom edge of the bright sky, not the true sky/ground boundary. The runway adds further bright outliers below the horizon that tug least-squares around.

2. **"Sky is on top of the image."** The column-scan looks top-down for the first ground pixel. Rotations break this catastrophically:
   - **sample3 (180°)** — every column's first row is already ground → `argmax` returns 0 for every column → line pinned at y=0.
   - **sample2, sample4 (~90°)** — the horizon runs vertically; most columns are all-sky or all-ground, so only a handful of "transitional" columns survive. The fit is noisy (sample2) or collapses (sample4).

3. **"No outlier is strong enough to bias the line."** Plain least-squares has zero outlier tolerance. Even if assumptions 1 and 2 held, a single tall tree or a bright runway patch can measurably shift the fit.

## Next-step recommendations

Ranked by effort vs. expected payoff.

1. **Rotation-invariant boundary extraction.** Instead of column-scan, extract the sky/ground boundary directly (e.g. fit via PCA on the binary mask, or trace the largest contour between regions and fit a line through it). Addresses failure mode 2 entirely. Small code delta.

2. **RANSAC / Theil–Sen line fit.** Replaces `np.polyfit`. Handles outliers from misclassified runway/tree patches. Addresses failure mode 3. One-line change if using `skimage.measure.ransac` or a hand-rolled 20-line RANSAC.

3. **Move beyond brightness-only classification.** Use color (HSV / Lab) or implement Ettinger's covariance-based method, which scores candidate lines by how well they separate pixels into two *colour-coherent* groups. Addresses failure mode 1. Larger lift (~80 lines), but standard classical UAV baseline in the literature.

A good attempt 2 = (1) alone. Attempt 3 = add (2). Attempt 4 = swap to (3) if accuracy still isn't enough.

## Open items (not blockers for attempt 1)

- ~~No ground-truth annotations on the samples → accuracy is eyeballed, not measured.~~ *Resolved:* the Horizon-UAV dataset ships `label.csv` + per-image masks; see the scoreboard above. Our four starter samples still need manual labels if we want per-sample numerical scores (especially for the rotated cases not represented in the dataset).
- Timings are on a dev laptop, not on the Raspberry Pi 5. Need to re-benchmark on target hardware before trusting the "≥15 FPS" claim.
