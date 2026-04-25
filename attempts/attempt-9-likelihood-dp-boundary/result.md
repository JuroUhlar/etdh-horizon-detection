# Attempt 9 - Likelihood-DP Boundary Candidate

This attempt keeps attempt 8's Otsu/RANSAC candidate pool, temporal prior, and
calibrated `no_horizon` abstention. The new piece is a conservative
low-resolution colour-likelihood boundary source:

1. Downsample the frame to `96x72`.
2. Build simple Lab/HSV/blue-excess features.
3. Estimate robust diagonal colour models from the top sky seed band and bottom
   ground seed band.
4. Use prefix sums plus dynamic programming to choose a smooth boundary path.
5. Fit a Huber line to that path and add it to the existing candidate pool only
   if its Ettinger coherence is at least `5.0` and its roll is at most `25 deg`.

The final gate matters. An ungated DP candidate regressed both datasets by
winning on weak no-horizon/treeline frames and on several steep UAV colour
splits. With the gate, the DP source only fires on a small number of strong
near-horizontal splits.

## Full-Dataset Results

Both final runs use `--seed 0` and were executed inside the Docker/RPi-budget
environment. Source of truth:
`full-eval-results-horizon_uav_dataset.json` and
`full-eval-results-video_clips_fpv_atv.json`.

### Horizon-UAV (480x480, 490 frames)

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Angle error (deg) | 0.97 | 0.75 | 2.02 | 6.16 |
| Position error (rho / H) | 0.0183 | 0.0154 | 0.0309 | 0.5439 |
| Hesse distance (px) | 8.80 | 7.39 | 14.85 | 261.09 |
| Sky-mask IoU | 0.885 | 0.951 | 0.980 | 0.997 |
| Latency, Docker/RPi model (ms) | 27.28 | 26.35 | 29.52 | 122.76 |

Pass rate: **476 / 490 = 97.14%**. Speed gate: PASS
(mean and p90 latency <= 66.7 ms).

This is +2 frames over attempt 8 and +1 frame over attempt 7 on Horizon-UAV.
The gains came from DP candidates on strong colour splits that the Otsu-derived
candidate pool missed.

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
| Latency, Docker/RPi model (ms) | 35.03 | 33.46 | 37.89 | 130.03 |

Pass rate: **69 / 120 = 57.50%**. Speed gate: PASS
(mean and p90 latency <= 66.7 ms).

This ties attempt 8 on FPV/ATV. The DP source does not fix the dominant
treeline/canopy failures; the strict admission gate mostly keeps it out of those
frames.

## Failed Sub-Experiments

The first version let a strong DP path override the early degenerate-mask
abstention. On the first 60 FPV frames it added two no-horizon false positives
and did not improve any line-scored horizon frames, so the early abstention was
restored.

The second version admitted every DP candidate that passed only seed/path
confidence. That recovered two UAV frames but also lost four UAV frames and one
FPV no-horizon frame. The final version gates DP candidates by the same
Ettinger score used for the rest of the pool plus a roll cap.

## Bottom Line

Attempt 9 is the best Horizon-UAV result so far while preserving attempt 8's
best FPV/ATV score. The cost is about 6 ms extra mean latency under Docker on
UAV and about 8.5 ms on FPV, still comfortably inside the 15 FPS budget. The
remaining FPV failures are still semantic treeline/canopy cases, not a lack of a
smooth colour boundary.
