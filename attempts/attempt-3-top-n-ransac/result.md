# Attempt 3 — Top-N RANSAC Candidates + Huber Refit

This attempt keeps the same mask-generation step as attempt 2, but upgrades the line search again.

The core idea is:

"Do not trust one line fit right away. Try many possible lines, see which boundary pixels agree with each one, group the strongest candidates, then refit the winner cleanly."

That makes the result much more stable when the boundary contains noise, clutter, or multiple plausible straight segments.

## Technique in Plain English

1. Build the same sky/ground mask as attempt 2.
2. Extract the boundary pixels between sky and ground.
3. Randomly sample many two-point lines from those boundary pixels.
4. Count how many boundary pixels support each sampled line.
5. Group overlapping good candidates together.
6. Refit each group with `cv2.fitLine` using Huber loss.
7. Rank the results by confidence and return the strongest one by default.

This is why the attempt is called "top-N RANSAC":

- **RANSAC**: repeatedly try random candidate lines and keep the ones with the most support.
- **Top-N**: the detector can keep several strong candidates, not just one.

For evaluation, the default detector call returns the best candidate only.

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
.venv/bin/python tools/evaluate.py attempts/attempt-3-top-n-ransac
```

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Δθ (angle error, deg) | 1.067 | 0.737 | 2.171 | 7.560 |
| Δρ (line position error, px) | 10.102 | 7.007 | 14.007 | 260.127 |
| Δρ / H (normalised line position error) | 0.021 | 0.015 | 0.029 | 0.542 |
| Sky-mask IoU | 0.929 | 0.952 | 0.984 | 0.997 |
| Latency (ms) | 69.553 | 41.805 | 156.164 | 384.337 |

**Pass rate:** `470 / 490 = 95.9%`

## Improvement Over Attempt 2

| Metric | Attempt 2 | Attempt 3 | Change |
|---|---:|---:|---:|
| Pass rate | 81.2% | 95.9% | +14.7 pts |
| Mean Δθ | 7.313° | 1.067° | much better |
| Mean Δρ | 36.700 px | 10.102 px | much better |
| Mean latency | 3.689 ms | 69.553 ms | much slower |

## What These Numbers Mean

- This is the most accurate attempt so far by a wide margin.
- The pass rate reaches `95.9%`, which is strong on this dataset.
- The angle error is now very low even in the tail: P90 `Δθ` is only `2.171°`.
- The sky-mask IoU stays essentially unchanged from attempt 2, which tells us the gain came from the candidate search and refit strategy, not from a better mask.
- The tradeoff is speed. This is far slower than attempts 1 and 2.

In plain terms: attempt 3 is much better at finding the right straight line, but you pay for that accuracy with runtime.

## Main Failure Modes

- **Speed**: RANSAC plus clustering plus refitting is much heavier than a single fit.
- **Shared mask limitations**: because the mask step is still the same as attempt 2, scenes with fundamentally wrong sky/ground separation can still fail.
- **Some large outliers remain**: max `Δρ / H` is still `0.542`, so a few frames are still badly off.

## Bottom Line

Attempt 3 is the accuracy leader. If the priority is benchmark score on this dataset, this is the current best attempt. If the priority is lightweight runtime, the cost is substantial and needs to be judged against the deployment target.
