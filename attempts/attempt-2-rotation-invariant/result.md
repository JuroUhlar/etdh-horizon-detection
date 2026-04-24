# Attempt 2 — rotation-invariant boundary extraction + robust line fit

Purpose: fix failure mode 2 (rotation) and partially address failure mode 3 (outlier bias) from attempt 1, while keeping the same sky/ground classifier.

## Method

1. **Grayscale + Gaussian blur (5×5) + Otsu threshold + morphological close/open (9×9 rect).** Identical to attempt 1. Otsu is orientation-invariant — it only looks at the brightness histogram — so it didn't need changing.
2. **Boundary extraction via morphological gradient (3×3).** Dilation − erosion of the sky mask produces a 1–2 pixel ridge along the sky/ground boundary in any orientation. Replaces the column-scan.
3. **Zero out the image frame border.** Where the sky region touches the edge of the image, the gradient reports a "boundary" that is actually the frame edge, not the horizon — dropping the 1-pixel border removes those spurious points.
4. **`cv2.fitLine` with Huber loss.** Orthogonal-distance regression over the boundary points. Returns a direction vector `(vx, vy)` and a point `(x0, y0)` on the line.
5. **Convert to (angle, y-intercept) for reporting.**
   - `angle_deg = atan2(vy, vx)`, normalised to `(-90, 90]`.
   - `intercept_y_at_x=0 = y0 − (vy/vx)·x0`, or `NaN` if `|vx|` is near zero (near-vertical line).
6. **Draw the line using the direction vector.** Works at any orientation, including vertical, where the y-intercept form would blow up.

## Why these changes

| Attempt 1 choice | Problem | Attempt 2 replacement |
|---|---|---|
| Per-column `argmax` on inverted mask | Assumes "sky is on top" — breaks on rotations | Morphological gradient → orientation-agnostic boundary points |
| `np.polyfit(x, y, 1)` | Minimises *vertical* distance; undefined for vertical lines; no outlier tolerance | `cv2.fitLine(..., DIST_HUBER)` — orthogonal distance + capped outlier influence |

## Parameters

| Param | Value | Rationale |
|---|---|---|
| Blur kernel | 5×5 Gaussian | Unchanged from attempt 1 |
| Threshold | Otsu (auto) | Unchanged |
| Closing + opening kernel | 9×9 rect | Added `MORPH_OPEN` pass to remove isolated specks that would become noise boundary points |
| Gradient kernel | 3×3 rect | Thinnest meaningful boundary; larger kernels widen the ridge without adding information |
| Line fit cost | `DIST_HUBER` | Robust to modest outliers; defaults to sensible Huber threshold when `param=0` |
| Fit precision | `reps=0.01, aeps=0.01` | OpenCV defaults; line convergence in radius (px) and angle (rad) |

## How to reproduce

```bash
for i in 1 2 3 4; do
  .venv/bin/python attempts/attempt-2-rotation-invariant/horizon_detect.py \
    "data/samples/sample${i}.jpg" \
    --out "attempts/attempt-2-rotation-invariant/outputs/sample${i}_horizon.jpg"
done
```

## Results

| Sample | Angle (deg) | Offset (px) | Time (ms) | Verdict |
|---|---:|---:|---:|---|
| sample1.jpg | −0.52 | +287.8 | 13.1 | Angle correct; line sits ~60 px *below* the true horizon (haze misclassified — failure mode 1) |
| sample2.jpg | +89.51 | vertical | 3.4 | Line lies on the real (vertical) sky/ground boundary |
| sample3.jpg | −0.52 | +195.5 | 3.6 | Line lies on the (inverted) sky/ground boundary |
| sample4.jpg | +89.51 | vertical | 3.5 | Line lies on the real (vertical) sky/ground boundary |

Note on "offset = vertical": for near-vertical horizons (|vx| < 1e-6), the y = m·x + b intercept is not finite. The raw direction vector `(vx, vy, x0, y0)` is retained in the detector's output and used for drawing, so the annotated image is always correct even when the numeric intercept isn't.

### Comparison to attempt 1

| Sample | Attempt 1 | Attempt 2 | Change |
|---|---|---|---|
| sample1 | −20.43° / 342.5 px | −0.52° / +287.8 px | Angle accurate; offset improved (~60 px off vs. ~100+ px off diagonally) |
| sample2 | +45.44° / −58.6 px | +89.51° / vertical | Degenerate → correct vertical fit |
| sample3 | 0° / 0 px (pinned to top) | −0.52° / +195.5 px | Degenerate → correct horizontal fit |
| sample4 | 0° / 0 px (pinned to top) | +89.51° / vertical | Degenerate → correct vertical fit |

## Full-dataset scoreboard (Horizon-UAV, 490 images)

Numerical evaluation against the mask-derived `label.csv` from the Horizon-UAV dataset, using `tools/evaluate.py`. Metrics are defined in [`docs/evaluation-metrics.md`](../../docs/evaluation-metrics.md).

| Metric | Attempt 1 | **Attempt 2** | Change |
|---|---:|---:|---|
| Δθ mean | 10.46° | **7.31°** | −30% |
| Δθ P50 | 1.41° | **0.92°** | sub-degree on median |
| Δθ P90 | 36.79° | **32.04°** | tail still rough |
| Δθ max | 85.06° | 88.74° | worst case slightly worse |
| Δρ / H mean | 14.7% | **7.6%** | halved |
| Δρ / H P50 | 2.3% | **1.6%** | — |
| Δρ / H P90 | 57.3% | **21.2%** | tail materially tighter |
| Sky-mask IoU mean | 0.926 | 0.929 | unchanged (same classifier) |
| Latency mean | 0.77 ms | 3.57 ms | +2.8 ms (still ~280 FPS on dev machine) |
| **Pass rate** (Δθ<5° & Δρ/H<5%) | 62.4% | **81.2%** | **+18.8 pp** |

Reproduce with:

```bash
.venv/bin/python tools/evaluate.py attempts/attempt-2-rotation-invariant
```

### What these numbers tell us

- **Pass rate is up 18.8 percentage points.** The rotation-invariant boundary extraction + Huber line fit recover a large fraction of the easy-to-medium cases that attempt 1's column-scan botched.
- **Sky-mask IoU is unchanged.** Moving from 0.926 to 0.929 is statistically nothing — the *classifier* (shared between attempts 1 and 2) didn't change. This is the clean signal that all improvements in attempt 2 came from the line-fit step alone. It's also a preview of why we can't go much further without changing the classifier: IoU can't go up until the sky mask does.
- **Median is sub-degree.** On typical samples the fit is within ~1° of ground truth.
- **Tail is still catastrophic** and substantially unchanged: max Δθ goes 85° → 89°. Attempt 2 can't help when the classifier hands it a mask whose boundary doesn't correspond to the real horizon.

### Worst-case analysis

The three worst offenders (annotated predictions saved in [`outputs/worst_cases/`](outputs/worst_cases/)) are the same three scenes that stumped attempt 1 — consistent with the shared-classifier theory:

| Image | Δθ | IoU | Root cause (classifier-level) |
|---|---:|---:|---|
| `a4-…_0.jpg` | 88.49° | 0.877 | High-roll overcast: sky (dull grey) and ground (dark greens) have near-identical luminance → Otsu splits along the wrong feature. Attempt 2 then fits cleanly to the *wrong* boundary. |
| `e6-…_140.jpg` | 88.74°/88.08°* | 0.705 | Sun glare on the right side of the frame is brighter than real sky. Otsu labels glare = sky, real sky = ground. Boundary tracks the glare, not the horizon. |
| `d6-…_140.jpg` | 88.08° | 0.707 | Near-duplicate of e6; same failure mode. |

*attempt 2's worst was actually `e6-…_160.jpg` (Δθ 88.74°); `_140.jpg` is 88.08°. Same clip, same failure mode.

**Pattern:** IoU 0.7–0.9 on the worst cases means Otsu got most of the pixels right, but the boundary it produced doesn't correspond to the real horizon. That's a pure classifier-failure signature — and the target for attempt 3.

## Failure modes still present

- **Failure mode 1 — haze misclassified as ground.** Sample 1's red line sits ~60 px below the true horizon because Otsu's cutoff lands at the bottom of the *bright* upper sky, treating the washed-out haze band as ground. Huber doesn't help: the haze produces a *coherent cluster* of incorrect boundary points, not isolated outliers. This genuinely needs a smarter sky/ground classifier.
- **Angle convention.** Outputs in (−90, 90] lose the distinction between "sky above" and "sky below". Not a problem for cropping (we can infer the sky side from the mask), but worth noting if a downstream consumer expects a signed attitude.

## Next-step recommendations

1. **Upgrade the classifier — attempt 3.** Replace Otsu-on-grayscale with either:
   - **Colour-based thresholding** (e.g. split on Lab's b* channel, where sky tends to skew blue and ground to skew yellow/green), or
   - **Ettinger's covariance method** — for each candidate line, score how well it separates pixels into two colour-coherent groups. Slower but highly robust; the classical UAV baseline in the literature.
   
   Either should fix sample 1's haze issue. Keep the rotation-invariant boundary fitter from this attempt as the downstream stage.

2. ~~**Get ground-truth annotations.** Every claim above is eyeballed.~~ *Resolved above.* The Horizon-UAV dataset gives us 490 labelled images and a numerical scoreboard. The four starter samples still lack per-sample labels and would need manual annotation (especially the rotated cases, which aren't represented in the upstream dataset).

3. **Benchmark on Raspberry Pi 5.** Timings here (3–13 ms on an M-series Mac) don't tell us anything about the ≥15 FPS target on ARM. Worth a pass early to spot any OpenCV build / BLAS issues before investing in a heavier classifier.
