# Evaluation metrics for horizon detection

How we compare a detector's output against ground truth, and why we deviate slightly from the hackathon's native parameterization for *scoring* purposes.

## Context

The hackathon spec describes the horizon as `y = m·x + b` and asks for the slope (or angle `θ = atan(m)`) and the vertical offset `b`. That's a fine *reporting* format — but it's a poor *comparison* format, because `b` goes undefined as the horizon approaches vertical.

In our four samples this is not a theoretical concern:
- Attempt 2, sample 2 (~90° rotated image): predicted `b ≈ −22 170 px`.
- Attempt 2, sample 4 (~90° rotated image): predicted `b ≈ −33 287 px`.

Both detections are *visually correct* — the line sits on the true vertical boundary — yet their `b` values are orders of magnitude apart and meaningless. Any metric of the form `|b_pred − b_gt|` would produce nonsense for these cases.

## Decision

**Compare lines in Hesse normal form `(θ, ρ)`**, but keep reporting the hackathon's `(angle, y-intercept)` when it's well-defined.

### Hesse normal form, briefly

Every line in the plane can be written as

```
x·cos(θ) + y·sin(θ) = ρ
```

where:

- **θ** ∈ (−90°, +90°] — the angle of the line (same as the detector's angle output, modulo 180°).
- **ρ** ∈ ℝ — the *signed perpendicular distance from the origin to the line*, in pixels.

Both values are finite at every orientation, including vertical — there's no singularity.

Given our detector's `(vx, vy, x0, y0)` output from `cv2.fitLine`, conversion is:

```python
theta = atan2(vy, vx)                  # direction of the line
# Normal to the line = (-sin(theta), cos(theta)); project any point on the
# line onto the normal to get rho.
rho = -sin(theta) * x0 + cos(theta) * y0
```

(Sign and the exact normal convention are a choice; we fix one and apply it consistently to both the prediction and the ground truth.)

### Metrics

| Metric | Definition | Units | Notes |
|---|---|---|---|
| **Angular error** | `min(|Δθ|, 180° − |Δθ|)` | degrees | Wraps mod 180° so that +89° vs. −89° counts as 2°, not 178° |
| **Positional error** | `|Δρ|` | pixels | Always finite. Intuitively: "how far apart are the two parallel copies of these lines?" |
| **(Bonus) Mask IoU** | `intersection / union` of predicted sky mask and ground-truth sky mask | unitless (0..1) | Available only when GT includes a mask, not just two points |
| **Latency** | wall-clock inference time | milliseconds | Reported but not error-bounded yet; target is ≥15 FPS on Raspberry Pi 5 |

### Reporting format

For *human-readable output* we still print angle and y-intercept (hackathon style). When the line is near-vertical (`|vx| < 1e-6`), we print `offset=vertical` and rely on the annotated image plus the Hesse-form metrics to convey correctness.

For *aggregated scoring* we always use `(θ, ρ)` errors and optional IoU.

## Aggregate statistics

Per-sample errors are aggregated into:

- **Mean angular error** (degrees) — average across the test set.
- **P50 / P90 / max angular error** — tail behaviour matters for UAVs; a 99%-of-the-time-correct detector can still crash.
- **Mean positional error** (pixels, normalised by image height for cross-resolution comparison).
- **Mean IoU** when masks are available.
- **Pass rate** — percentage of frames within a tolerance (e.g. `|Δθ| < 5°` *and* `|Δρ|/H < 5%`).

## Why not just use IoU?

IoU is a great metric when we have ground-truth masks — it does not depend on a straight-line assumption and handles complex horizons (mountains, buildings) naturally. But the hackathon explicitly asks for a line, not a mask, so we need line-level metrics too. In practice we'll report both when masks are available: IoU shows *how well we segmented*, `(Δθ, Δρ)` show *how well we extracted a line from that segmentation*.

## Ground truth sources

Two complementary paths, both feeding into the same evaluator.

1. **Manual two-point annotation** — for quick labelling of our four starter samples. Click the left and right ends of the horizon in a tiny tool; a JSON of `{image: [(x1, y1), (x2, y2)]}` falls out. Converts trivially to `(θ, ρ)`.
2. **Mask-derived annotation** — for the ~500-image Horizon-UAV dataset. Extract the sky/land mask boundary, fit a robust line (same `cv2.fitLine` trick we use in the detector), store `(θ, ρ)` as GT. Also retain the mask so we can compute IoU directly.

Both produce the same downstream JSON shape:

```json
{
  "sample1.jpg": { "theta_deg": 0.5, "rho_px": 141.2, "image_size": [480, 480] },
  "sample2.jpg": { "theta_deg": 89.8, "rho_px": 95.3, "image_size": [480, 480] }
}
```

so the evaluator doesn't care which source produced them.

## Open items

- **Coordinate convention.** Image coordinates have y pointing *down*; common maths references have y pointing *up*. The sign of θ and ρ depends on this choice. We will fix image-coords (y down, positive angles rotate the +x axis toward +y = clockwise) and enforce it in both the GT extractor and the evaluator.
- **Mask IoU on near-vertical lines.** For rotated samples, the "sky" side label in the mask may not align with whichever side the detector labelled as sky. The IoU computation needs to try both assignments and take the better one, otherwise we'll score ~0 on correctly-detected flipped cases.
- **Test-set curation.** If we ingest the Horizon-UAV dataset, we should still keep a small held-out set of our own samples (and any custom captures) so numbers reflect generalisation, not overfit-to-one-dataset.
