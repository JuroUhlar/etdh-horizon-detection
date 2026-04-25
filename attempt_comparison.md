# Attempt Comparison

This file compares the three horizon-detection attempts in plain language.

## Metric Labels

- **Pass rate**: the share of images where the prediction is close enough in both angle and position. In this repo that means `Δθ < 5°` and `Δρ / H < 5%`.
- **Δθ**: angle error. How many degrees the predicted horizon is tilted away from the ground-truth horizon.
- **Δρ**: line position error in pixels. How far the predicted line is shifted from the ground-truth line.
- **Δρ / H**: the same line position error, divided by image height so the number is comparable across image sizes.
- **Sky-mask IoU**: overlap between the predicted sky region and the ground-truth sky region. `1.0` is perfect.
- **Mean latency**: average runtime per image.

Smaller is better for `Δθ`, `Δρ`, `Δρ / H`, and latency. Bigger is better for pass rate and IoU.

## Techniques, In Plain English

| Attempt | Main idea | Strength | Weakness |
|---|---|---|---|
| Attempt 1 | Split the frame into bright sky and darker ground, then scan each column from top to bottom and fit one line | Very fast and simple | Breaks on rotated scenes and brittle boundaries |
| Attempt 2 | Keep the same mask, but find the boundary in a rotation-invariant way and fit a more robust line | Big accuracy jump for small extra cost | Still depends on the same brightness-based mask |
| Attempt 3 | Keep the same mask, but try many candidate lines, keep the ones with the most support, and refit the winner | Best line accuracy by far | Much slower |

## Full Results

All numbers below are from:

```bash
.venv/bin/python tools/evaluate.py <attempt-dir>                                    # default: Horizon-UAV
.venv/bin/python tools/evaluate.py <attempt-dir> --dataset data/video_clips_ukraine_atv
```

The two datasets stress very different things, so we report them side by side rather than averaging — the takeaways are not the same.

### Horizon-UAV (`490` images, 480×480, every frame has a horizon)

| Metric | Attempt 1 | Attempt 2 | Attempt 3 |
|---|---:|---:|---:|
| Pass rate | 62.4% | 81.2% | 95.7% |
| Mean Δθ | 10.461° | 7.313° | 1.113° |
| P50 Δθ | 1.415° | 0.917° | 0.734° |
| P90 Δθ | 36.793° | 32.036° | 2.166° |
| Max Δθ | 85.059° | 88.744° | 28.181° |
| Mean Δρ | 70.744 px | 36.700 px | 10.458 px |
| Mean Δρ / H | 0.147 | 0.076 | 0.022 |
| Mean Sky-mask IoU | 0.926 | 0.929 | 0.929 |
| Mean latency | 0.639 ms | 3.579 ms | 69.703 ms |

Attempt 3's mean Δθ wobbles slightly between runs (it's the only stochastic attempt — RANSAC's hypothesis sampling is seeded per frame but not across the run), so re-running is expected to give numbers within ~0.05° of these.

### Ukraine ATV FPV (`120` labelled frames, 1920×1080, 110 horizon + 10 no-horizon)

| Metric | Attempt 1 | Attempt 2 | Attempt 3 |
|---|---:|---:|---:|
| Pass rate | 3.3% | 0.0% | 0.0% |
| Mean Δθ (TP frames only) | 14.039° | 13.900° | 82.802° |
| P50 Δθ | 12.865° | 13.237° | 86.339° |
| P90 Δθ | 27.950° | 26.581° | 89.289° |
| Max Δθ | 50.613° | 47.935° | 89.992° |
| Mean Δρ | 361.246 px | 362.192 px | 1104.496 px |
| Mean Δρ / H | 0.334 | 0.335 | 1.023 |
| Mean latency | 4.125 ms | 34.973 ms | 562.579 ms |
| Confusion matrix (TP / FN / FP / TN) | 110 / 0 / 10 / 0 | 110 / 0 / 10 / 0 | 110 / 0 / 10 / 0 |

A few things to read carefully here:

- **`Mean Δθ` is averaged over TP frames only** — the 110 frames where both the label and the detector say there's a horizon. The 10 no-horizon frames don't have a ground-truth line, so line errors are not defined for them.
- **Pass rate is over all 120 frames.** The 10 no-horizon frames automatically fail because every attempt currently emits a line on every frame (FP=10, TN=0 in the confusion matrix). Even if line accuracy on the 110 TP frames were perfect, the ceiling would be 91.7%.
- **Δρ / H > 1** for attempt 3 means the predicted line is, on average, *more than one image-height of distance* from the ground-truth line in Hesse normal form. Combined with mean Δθ near 83°, this is not "slightly off" — it's "perpendicular to the truth", which is what RANSAC produces when it locks onto the wrong dominant edge.
- **Latency on ATV is 5–8× the UAV latency** for each attempt, because ATV frames are 1920×1080 (~9× the pixel area of the 480×480 UAV frames). All three pipelines run at full resolution, so cost scales roughly linearly with pixel count.

## What Changed From Attempt To Attempt (Horizon-UAV)

### Attempt 1 -> Attempt 2

- Rotation handling improved a lot.
- Pass rate improved from `62.4%` to `81.2%`.
- Mean line-position error roughly halved.
- Mean latency increased from `0.639 ms` to `3.579 ms`, which is still very cheap on the dev machine.

Interpretation:

Attempt 2 shows that the first big problem was not only the mask. A lot of the error in attempt 1 came from the way the boundary was extracted and the way the line was fitted.

### Attempt 2 -> Attempt 3

- Pass rate improved again, from `81.2%` to `95.7%`.
- Mean angle error dropped sharply from `7.313°` to `1.113°`.
- Mean line-position error dropped from `36.700 px` to `10.458 px`.
- Mean latency jumped from `3.579 ms` to `69.703 ms`.

Interpretation:

Attempt 3 shows that the strongest remaining gains came from a better search over possible lines, not from a better sky mask. That is why IoU stays almost unchanged while the line metrics improve dramatically.

## Why The Story Inverts On Ukraine ATV

The Horizon-UAV table tells a "monotone improvement" story. The ATV table does not — attempt 3 is the *worst* of the three on that data, and by a wide margin. That is the most important finding from adding the second dataset.

All three attempts share the same first stage: an Otsu-style brightness threshold that splits the frame into "bright = sky" and "dark = ground", followed by a boundary-extraction step and a line fit. They differ only in how they extract the boundary and how they fit the line.

On Horizon-UAV that mask is mostly correct, because the upstream dataset is dominated by clear-sky aerial scenes where sky really is brighter than ground. On Ukraine ATV FPV footage that assumption breaks: low-altitude treeline shots, road approaches, and ground-POV frames have no clean brightness split. The boundary the mask produces is not the horizon — it's whatever bright/dark contour happens to win Otsu.

Once that first stage is wrong, each downstream choice amplifies the error differently:

- **Attempt 1** scans columns top-to-bottom on the bad mask, then fits one line by least squares. The fit averages over noise and ends up "wrong-but-roughly-horizontal" — `Δθ ≈ 14°`. Bad, but not catastrophically so.
- **Attempt 2** improves the boundary and fit, but on the same bad mask the rotation-invariant fit lands in roughly the same place — `Δθ ≈ 13.9°`. The rotation invariance is not buying anything here because the mask itself is what's wrong.
- **Attempt 3** runs RANSAC over candidate lines on the same bad mask and picks the one with the most inlier support. On FPV footage the strongest line in the boundary set is often a *vertical* artefact — tree trunks, fence edges, road edges — and RANSAC commits to it with high confidence. The mean Δθ of 82.8° is what "RANSAC is sure, and RANSAC is wrong" looks like.

In short: a more aggressive line search is a force multiplier. When the underlying signal is right, it multiplies you toward the truth (UAV). When the underlying signal is wrong, it multiplies you away from it (ATV).

The 10 no-horizon frames are a separate failure orthogonal to all of this. None of the three attempts implements the `no_horizon` return path that the evaluator supports, so they take a hard 10/120 hit on classification before line accuracy is even measured.

## Recommended Reading Of The Results

- If you care most about **raw accuracy on the Horizon-UAV benchmark**, attempt 3 is the clear winner.
- If you care most about **speed**, attempt 1 is the winner, but it is not reliable enough — and on FPV-style footage no current attempt is reliable.
- If you care about **best balance of simplicity, speed, and improvement on UAV**, attempt 2 is the most practical middle ground.
- If you care about **robustness across both datasets**, none of the three is acceptable yet. The shared brightness-mask first stage is the binding constraint, not the line-fitting strategy.

## Practical Takeaway

The two datasets together tell a more honest story than either alone:

1. A naive brightness split plus simple line fit is not reliable enough on UAV, and is fundamentally wrong on FPV.
2. Making the boundary extraction and fit rotation-invariant solves a large class of UAV failures, but does not help when the mask itself is not a horizon.
3. Searching across many candidate lines is the strongest UAV improvement, but it is also the riskiest choice when the mask is bad — high-confidence support for the wrong line is worse than uncertain support for it.
4. The next meaningful improvement is not "fit lines better" — it's "stop trusting the brightness mask as a proxy for the horizon", and "let the detector abstain when no horizon is present".
