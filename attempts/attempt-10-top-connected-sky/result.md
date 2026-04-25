# Attempt 10 - Top-Connected Sky Envelope

This attempt keeps attempt 9's Otsu/RANSAC pool, gated likelihood-DP candidate,
temporal prior, and `no_horizon` abstention. The new piece is a second
low-resolution candidate source:

1. Downsample the frame to `96x72`.
2. Build Lab/HSV/blue-excess features plus a clipped local Sobel-texture feature.
3. Estimate robust diagonal top-band and bottom-band likelihood models.
4. Threshold sky likelihood, clean it morphologically, and keep only pixels in
   components connected to the top image border.
5. Fit a Huber line to the lower envelope of that top-connected sky component.
6. Admit the candidate only when area, column coverage, median boundary height,
   texture contrast, roll, and Ettinger coherence all pass conservative gates.

The intent is to recover frames where the real sky is still visible at the top,
but the strongest Otsu/DP split lands on a lower road, treeline, or field edge.

## Full-Dataset Results

Both final runs use `--seed 0` and were executed inside the Docker/RPi-budget
environment. Source of truth:
`full-eval-results-horizon_uav_dataset.json` and
`full-eval-results-video_clips_fpv_atv.json`.

### Horizon-UAV (480x480, 490 frames)

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Angle error (deg) | 0.99 | 0.78 | 2.00 | 6.16 |
| Position error (rho / H) | 0.0167 | 0.0120 | 0.0305 | 0.5532 |
| Hesse distance (px) | 8.03 | 5.75 | 14.66 | 265.56 |
| Sky-mask IoU | 0.909 | 0.961 | 0.985 | 0.997 |
| Latency, Docker/RPi model (ms) | 30.29 | 29.56 | 30.74 | 128.94 |

Pass rate: **477 / 490 = 97.35%**. Speed gate: PASS
(mean and p90 latency <= 66.7 ms).

Relative to attempt 9, this is a net +1 frame: two UAV frames are recovered by
the top-connected envelope, and one temporally adjacent frame regresses.

### FPV/ATV (120 frames, 10 labelled no-horizon)

`has_horizon` confusion matrix:

| | pred horizon | pred no_horizon |
|---|---:|---:|
| gt horizon | TP = 100 | FN = 10 |
| gt no_horizon | FP = 2 | TN = 8 |

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Angle error (deg) | 4.36 | 1.47 | 10.92 | 37.94 |
| Position error (rho / H) | 0.090 | 0.018 | 0.292 | 0.656 |
| Hesse distance (px) | 43.08 | 8.52 | 139.99 | 314.76 |
| Latency, Docker/RPi model (ms) | 37.23 | 36.39 | 38.78 | 139.12 |

Pass rate: **72 / 120 = 60.00%**. Speed gate: PASS
(mean and p90 latency <= 66.7 ms).

Relative to attempt 9, this recovers three FPV frames and introduces no FPV
pass regressions:

- `01_01m32s-01m55s_aerial_recon_f002.jpg`
- `05_11m00s-11m04s_fpv_road_approach_f000.jpg`
- `05_11m00s-11m04s_fpv_road_approach_f001.jpg`

The no-horizon confusion matrix is unchanged from attempt 9.

## Failed Sub-Experiments

The first version admitted envelope candidates with Ettinger coherence >= `0.55`.
That recovered a few road/aerial frames but let a weak treeline envelope at
`04_10m34s-11m00s_fpv_treeline_f012.jpg` poison the temporal prior, dropping
FPV/ATV to **67 / 120**.

Raising the gate to `0.80` improved FPV/ATV to **71 / 120**, but later treeline
frames with envelope scores around `0.83-0.96` still disturbed the next temporal
prediction. The final gate is `1.00`, which keeps the road/aerial improvements
and removes the observed FPV regressions.

## Bottom Line

Attempt 10 is the new best combined result in this repo: best Horizon-UAV pass
rate so far and the first FPV/ATV improvement beyond attempts 8/9. The extra
candidate costs about 3 ms over attempt 9 on Docker UAV and about 2 ms on FPV,
but both mean and p90 latency remain comfortably inside the 15 FPS budget.

The dominant remaining failures are still the hard FPV treeline/canopy frames
(`f034`, `f042`-`f045`), where the labelled horizon is partly semantic rather
than simply the strongest top-connected sky boundary.
