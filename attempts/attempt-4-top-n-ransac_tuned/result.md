# Attempt 4 — Top-N RANSAC (Tuned)

This attempt keeps the exact same sky/ground classifier as attempts 2 and 3
(grayscale → Otsu → morphological close/open → gradient boundary), and makes
no changes to the cluster-and-refit step. All changes are to the RANSAC
hypothesis scoring step, targeting latency without intentionally sacrificing
Horizon-UAV accuracy.

## What Changed From Attempt 3

### 1. Vectorised RANSAC hypothesis scoring

Attempt 3 scored each of many candidate lines in a Python `for` loop. Attempt 4
scores all hypotheses in a single matrix operation — candidate direction
vectors are stacked into a distance matrix and evaluated at once, removing
per-iteration Python overhead.

### 2. Boundary-point subsampling

Before RANSAC, boundary pixels are subsampled to at most 400 points. Dense
boundaries can have thousands of pixels; subsampling caps matrix size. RANSAC
is stochastic anyway, so this mainly affects cost, not the intended line.

### 3. Fewer RANSAC iterations (300 vs 500)

With a vectorised scorer, 300 iterations cost much less than 500 did in the
original loop, while the hypothesis pool remains large enough to find the
horizon in practice.

## Full-Dataset Results (Horizon-UAV)

Measured on the Horizon-UAV dataset (`490` images) with:

```bash
.venv/bin/python tools/evaluate.py attempts/attempt-4-top-n-ransac_tuned
```

`full-eval-results-horizon_uav_dataset.json` in this directory is the
machine-readable source of truth for the run that produced the table below.

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Δθ (angle error, deg) | 1.078 | 0.761 | 2.296 | 7.704 |
| Δρ (line position error, px) | 10.625 | 6.954 | 14.188 | 260.342 |
| Δρ / H (normalised line position error) | 0.022 | 0.014 | 0.030 | 0.542 |
| Sky-mask IoU | 0.929 | 0.952 | 0.984 | 0.997 |
| Latency (ms) | 18.006 | 10.871 | 39.781 | 132.850 |

**Pass rate:** `468 / 490 = 95.5%`

On the **FPV/ATV clip** set (`--dataset data/video_clips_fpv_atv`), see
`full-eval-results-video_clips_fpv_atv.json` — pass rate and mean errors
differ from Horizon-UAV because the brightness mask and no-horizon labels hit
harder; attempt 3 can still lead on line accuracy in some runs (stochastic
RANSAC).

**Speed budget (15 FPS ≈ 67 ms):** the evaluator’s aggregate verdict on this
run is pass on mean latency; a small fraction of frames exceed 67 ms on the
slow tail. Treat worst-case timing on real Pi hardware as the final gate, not
dev-machine numbers alone.

## Comparison With Previous Attempts (Horizon-UAV, same host run)

| Metric | Attempt 2 | Attempt 3 | Attempt 4 |
|---|---:|---:|---:|
| Pass rate | 81.2% | 95.5% | 95.5% |
| Mean latency | 3.70 ms | 71.5 ms | **18.0 ms** |
| P90 latency | 7.0 ms | 161.0 ms | **39.8 ms** |
| Worst latency | 20.4 ms | 393.5 ms | **132.8 ms** |

## Remaining Failure Modes

All remaining failures are **classifier failures**, not fitting failures — the
same root causes as attempts 2 and 3:

| Failure mode | Trigger |
|---|---|
| Luminance ambiguity | Overcast sky ≈ grey ground; Otsu cuts at the wrong level |
| Sun glare | Bright patch in sky labelled as ground by Otsu |
| Haze band | Washed-out haze darker than upper sky confuses the threshold |

A colour-based classifier (Lab b*, or Ettinger's covariance method) is the most
plausible next step to push the pass rate beyond the mid-90s on Horizon-UAV
while keeping runtime sane.

## Bottom Line

Attempt 4 matches attempt 3’s pass rate on Horizon-UAV in our latest evaluation
with roughly **4×** lower mean latency than attempt 3 on the same host, making
it the better default when both accuracy and CPU budget matter. The shared
Otsu mask and lack of a `no_horizon` path remain the main robustness limits on
other datasets.
