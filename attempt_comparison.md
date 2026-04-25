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
.venv/bin/python tools/evaluate.py <attempt-dir> --seed 0                          # pin stochastic detectors
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

Attempt 3 is the only stochastic attempt. If you run the commands above without `--seed`, its metrics will wobble slightly from run to run; pass `--seed 0` to pin an exact result when reproducing a table.

### Ukraine ATV FPV (`120` labelled frames, cropped + resized to ~625×480, 110 horizon + 10 no-horizon)

| Metric | Attempt 1 | Attempt 2 | Attempt 3 |
|---|---:|---:|---:|
| Pass rate | 16.7% | 4.2% | 45.0% |
| Mean Δθ (TP frames only) | 7.7° | 15.9° | 7.3° |
| Mean Δρ | 59.7 px | 100.0 px | 61.5 px |
| Mean Δρ / H | 0.124 | 0.208 | 0.128 |
| Mean latency | 0.9 ms | 30.9 ms | 754.7 ms |
| Confusion matrix (TP / FN / FP / TN) | 110 / 0 / 10 / 0 | 110 / 0 / 10 / 0 | 110 / 0 / 10 / 0 |

A few things to read carefully here:

- **`Mean Δθ` is averaged over TP frames only** — the 110 frames where both the label and the detector say there's a horizon. The 10 no-horizon frames don't have a ground-truth line, so line errors are not defined for them.
- **Pass rate is over all 120 frames.** The 10 no-horizon frames automatically fail because every attempt currently emits a line on every frame (FP=10, TN=0 in the confusion matrix). Even if line accuracy on the 110 TP frames were perfect, the ceiling would be 91.7%.
- **Cropping the black side bars changes the result materially.** Attempt 3 goes from "catastrophically wrong" to the clear accuracy leader on ATV once the frame-border artefacts are removed.
- **The no-horizon failure is unchanged.** All three attempts still emit a line on all 120 frames, so the ceiling remains 91.7% until the detector can abstain.
- **Latency no longer scales cleanly with pixel count.** Attempt 1 becomes much cheaper after resizing, attempt 2 changes only modestly, and attempt 3 gets slower because the resized frames produce a denser boundary point cloud for its RANSAC stage.

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

## What Changed On Ukraine ATV After Cropping

The original ATV result was dominated by the dataset itself: large black side bars created strong artificial edges and pulled the detectors, especially the RANSAC pipeline, toward the frame border instead of the horizon. Rewriting the ATV frames to remove those bars changes the story completely.

All three attempts share the same first stage: an Otsu-style brightness threshold that splits the frame into "bright = sky" and "dark = ground", followed by a boundary-extraction step and a line fit. They differ only in how they extract the boundary and how they fit the line.

On Horizon-UAV that mask is mostly correct, because the upstream dataset is dominated by clear-sky aerial scenes where sky really is brighter than ground. On Ukraine ATV FPV footage that assumption still breaks in many frames: treeline shots, road approaches, and ground-POV footage often have no clean brightness split. But after cropping the side bars, the boundary set is no longer polluted by a pair of huge vertical frame edges.

Once that first stage is wrong, each downstream choice amplifies the error differently:

- **Attempt 1** benefits immediately. Once the frame-border artefacts are gone, its simplistic column scan is much less likely to fit the border, so pass rate rises from `3.3%` to `16.7%`.
- **Attempt 2** does not benefit as much. It remains heavily constrained by the quality of the Otsu mask itself, so it is still often fitting the wrong boundary even though the most obvious artificial edges are gone.
- **Attempt 3** benefits the most in accuracy. Its aggressive search is no longer being handed an easy, dominant vertical border line, so it can often lock onto the real horizon. That is why pass rate jumps from `0.0%` to `45.0%` and mean Δθ falls from `82.8°` to `7.3°`.

In short: the more aggressive line search was not intrinsically wrong on ATV; it was being poisoned by bad input geometry. Once the dataset is cleaned up, RANSAC becomes useful again. The remaining failure mode is the brightness mask itself, plus the missing no-horizon path.

The 10 no-horizon frames are a separate failure orthogonal to all of this. None of the three attempts implements the `no_horizon` return path that the evaluator supports, so they take a hard 10/120 hit on classification before line accuracy is even measured.

## Recommended Reading Of The Results

- If you care most about **raw accuracy on the Horizon-UAV benchmark**, attempt 3 is the clear winner.
- If you care most about **speed**, attempt 1 is still the winner, but its ATV accuracy is still limited.
- If you care about **best balance of simplicity, speed, and improvement on UAV**, attempt 2 is the most practical middle ground.
- If you care about **robustness across both datasets**, attempt 3 is now the strongest line detector on both, but none of the three is acceptable yet because the shared brightness-mask first stage and the missing no-horizon classifier still cap robustness.

## Practical Takeaway

The two datasets together tell a more honest story than either alone:

1. A naive brightness split plus simple line fit is not reliable enough on UAV, and is fundamentally wrong on FPV.
2. Cropping away large artificial borders can matter as much as algorithm changes; dataset hygiene was part of the ATV failure.
3. Making the boundary extraction and fit rotation-invariant solves a large class of UAV failures, but it still does not help much when the mask itself is not a horizon.
4. Searching across many candidate lines is the strongest improvement once the input geometry is sane, but it is still vulnerable when the mask is bad.
5. The next meaningful improvement is not just "fit lines better" — it's "stop trusting the brightness mask as a proxy for the horizon", and "let the detector abstain when no horizon is present".
