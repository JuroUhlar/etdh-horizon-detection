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
.venv/bin/python tools/evaluate.py <attempt-dir>
```

on the Horizon-UAV dataset (`490` images).

| Metric | Attempt 1 | Attempt 2 | Attempt 3 |
|---|---:|---:|---:|
| Pass rate | 62.4% | 81.2% | 95.9% |
| Mean Δθ | 10.461° | 7.313° | 1.067° |
| P50 Δθ | 1.415° | 0.917° | 0.737° |
| P90 Δθ | 36.793° | 32.036° | 2.171° |
| Max Δθ | 85.059° | 88.744° | 7.560° |
| Mean Δρ | 70.744 px | 36.700 px | 10.102 px |
| Mean Δρ / H | 0.147 | 0.076 | 0.021 |
| Mean Sky-mask IoU | 0.926 | 0.929 | 0.929 |
| Mean latency | 0.762 ms | 3.689 ms | 69.553 ms |

## What Changed From Attempt To Attempt

### Attempt 1 -> Attempt 2

- Rotation handling improved a lot.
- Pass rate improved from `62.4%` to `81.2%`.
- Mean line-position error roughly halved.
- Mean latency increased from `0.762 ms` to `3.689 ms`, which is still very cheap on the dev machine.

Interpretation:

Attempt 2 shows that the first big problem was not only the mask. A lot of the error in attempt 1 came from the way the boundary was extracted and the way the line was fitted.

### Attempt 2 -> Attempt 3

- Pass rate improved again, from `81.2%` to `95.9%`.
- Mean angle error dropped sharply from `7.313°` to `1.067°`.
- Mean line-position error dropped from `36.700 px` to `10.102 px`.
- Mean latency jumped from `3.689 ms` to `69.553 ms`.

Interpretation:

Attempt 3 shows that the strongest remaining gains came from a better search over possible lines, not from a better sky mask. That is why IoU stays almost unchanged while the line metrics improve dramatically.

## Recommended Reading Of The Results

- If you care most about **raw accuracy on this dataset**, attempt 3 is the clear winner.
- If you care most about **speed**, attempt 1 is the winner, but it is not reliable enough.
- If you care about **best balance of simplicity, speed, and improvement**, attempt 2 is the most practical middle ground.

## Practical Takeaway

The attempts tell a clear story:

1. A naive brightness split plus simple line fit is not reliable enough.
2. Making the boundary extraction and fit rotation-invariant solves a large class of failures.
3. Searching across many candidate lines and choosing the strongest one is what gets the score from "pretty good" to "very strong", but it introduces a major runtime cost.
