# Attempt 8 - Temporal Prior + Calibrated Abstention

This attempt keeps attempt 7's successful per-frame detector:

1. Extract Otsu/morph boundary points from Lab L and Lab b*.
2. Run vectorised top-K RANSAC on both channels.
3. Pool candidates and rerank by Ettinger-style Lab region coherence.
4. Refit the winning inliers with `cv2.fitLine(..., DIST_HUBER)`.

The new part is a small video-stream prior. A 16x16 Lab-L thumbnail is used as
a scene-change check. If the current frame is continuous with the previous
accepted frame, candidate scores get a soft multiplicative boost when their
Hesse line is close to the previous accepted line. The raw coherence floor is
still enforced, so temporal memory can only choose among plausible current-frame
candidates; it cannot force a line through a frame whose evidence is too weak.

The abstention floor was also raised from `0.15` to `0.22`. A sweep on
FPV/ATV showed this was the best trade-off with the temporal prior: fewer
no-horizon false positives, at the cost of more false negatives.

## Pipeline Changes

```text
candidate_score =
    ettinger_score(candidate)
    * (1 + TEMPORAL_WEIGHT * temporal_prior(candidate, previous_line))

temporal_prior =
    exp(-((delta_theta / 8 deg)^2 + (delta_rho_norm / 0.14)^2))
```

Temporal state resets when the 16x16 thumbnail mean absolute difference is
above `22.0`, which keeps the prior active on most FPV sequence frames and
mostly inactive on the shuffled-looking Horizon-UAV ordering.

## Full-Dataset Results

Both final runs use `--seed 0` and were executed inside the Docker/RPi-budget
environment. Source of truth:
`full-eval-results-horizon_uav_dataset.json` and
`full-eval-results-video_clips_fpv_atv.json`.

### Horizon-UAV (480x480, 490 frames)

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Angle error (deg) | 1.01 | 0.75 | 2.08 | 8.20 |
| Position error (rho / H) | 0.0186 | 0.0154 | 0.0311 | 0.5439 |
| Hesse distance (px) | 8.95 | 7.40 | 14.91 | 261.09 |
| Sky-mask IoU | 0.884 | 0.951 | 0.980 | 0.997 |
| Latency, Docker/RPi model (ms) | 21.51 | 20.93 | 21.72 | 108.26 |

Pass rate: **474 / 490 = 96.73%**. Speed gate: PASS
(mean and p90 latency <= 66.7 ms).

Attempt 7 was 475 / 490, so this loses one UAV frame. The loss is from the
higher coherence floor and is the main cost of the FPV improvement.

### FPV/ATV (120 frames, 10 labelled no-horizon)

`has_horizon` confusion matrix:

| | pred horizon | pred no_horizon |
|---|---:|---:|
| gt horizon | TP = 100 | FN = 10 |
| gt no_horizon | FP = 2 | TN = 8 |

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Angle error (deg) | 4.83 | 1.87 | 11.69 | 37.94 |
| Position error (rho / H) | 0.102 | 0.018 | 0.328 | 0.656 |
| Hesse distance (px) | 48.93 | 8.77 | 157.36 | 314.76 |
| Latency, Docker/RPi model (ms) | 26.51 | 25.66 | 26.84 | 114.56 |

Pass rate: **69 / 120 = 57.50%**. Speed gate: PASS
(mean and p90 latency <= 66.7 ms).

Compared with attempt 7, FPV improves from 63 / 120 to 69 / 120 (+5.0 pp).
The confusion matrix shifts from `TP=105, FN=5, FP=4, TN=6` to
`TP=100, FN=10, FP=2, TN=8`: the higher floor correctly abstains on two more
no-horizon frames but also abstains on five more horizon frames. The remaining
net gain comes from temporal candidate selection reducing jitter on some
line-scored FPV frames.

## Remaining Failure Modes

The worst FPV failures are unchanged: `04_10m34s-11m00s_fpv_treeline_f034`,
`f042`, `f043`, `f044`, and `f045` still dominate the tail, with angle errors
up to 37.9 deg and position errors up to 65.6% of image height. These are not
random jitter; the visible scene and labels imply a different semantic boundary
than the strongest sky/ground colour split.

The temporal prior helps only when the correct candidate already exists in the
pooled RANSAC set. It does not solve cases where both threshold masks omit the
true boundary, and it intentionally refuses to hallucinate a line when raw
coherence is below the abstention floor.

## Failed Sub-Experiment

Before settling on the temporal prior, I tried adding a capped Canny +
probabilistic-Hough source as an independent candidate pool. On the first 60
FPV frames it regressed from 35 / 60 to 31 / 60 and increased false positives
on no-horizon frames. The long edges it found were often roads, overlays, or
tree/field structure rather than the labelled horizon. I reverted that path.

## Bottom Line

Attempt 8 is a modest FPV improvement that stays comfortably within the speed
budget, but it is not a breakthrough. It trades one UAV pass and five extra FPV
false negatives for fewer no-horizon false positives and better candidate
choice on some continuous FPV frames. The hardest treeline frames remain a
classifier/semantics problem rather than a fitter problem.
