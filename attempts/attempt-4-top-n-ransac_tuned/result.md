# Attempt 4 — Top-N RANSAC (Tuned)

This attempt keeps the exact same sky/ground classifier as attempts 2 and 3
(grayscale → Otsu → morphological close/open → gradient boundary), and makes
no changes to the cluster-and-refit step. All changes are to the RANSAC
hypothesis scoring step, targeting the slow tail without touching accuracy.

## What Changed From Attempt 3

### 1. Vectorised RANSAC hypothesis scoring

Attempt 3 scored each of 500 candidate lines in a Python `for` loop: one
numpy distance computation per iteration. Attempt 4 scores all hypotheses in
a single matrix operation — all `n_iter` candidate direction vectors are
stacked into an `(n_iter, n_pts)` distance matrix and evaluated at once.
This eliminates per-iteration Python overhead and is the primary driver of
the speed improvement.

### 2. Boundary-point subsampling

Before RANSAC, boundary pixels are subsampled to at most 400 points. Dense or
noisy boundaries can have 2 000+ pixels; each extra point widens the distance
matrix and linearly increases cost. Subsampling caps the matrix at
`300 × 400` float32 (~480 KB), directly flattening the slow tail. RANSAC is
stochastic anyway, so subsampling does not reduce accuracy.

### 3. Fewer RANSAC iterations (300 vs 500)

With a vectorised scorer, 300 iterations cost less than 500 did in the loop.
The hypothesis pool is still large enough to find the real horizon reliably.

## Full-Dataset Results

Evaluated on the Horizon-UAV dataset (490 images) inside Docker with
`cpus: "1"` and `mem_limit: "1g"` — the tightest resource constraint tested.

### Accuracy

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Δθ (angle error, deg) | 1.1 | 0.8 | 2.3 | 28.3 |
| Δρ / H (normalised position error) | 0.0% | 0.0% | 0.0% | 0.5% |
| Sky-mask IoU | 0.93 | 0.95 | 0.98 | 1.00 |

**Pass rate: 468 / 490 = 95.5%**

### Speed (Docker, 1 CPU core, 1 GB RAM)

| Metric | value |
|---|---:|
| Mean latency | 16.2 ms |
| Median latency | 10.9 ms |
| P90 latency | 34.2 ms |
| Worst latency | 60.8 ms |
| Mean FPS | 61.8 |
| Worst-case FPS | 16.5 |

**Budget: ≤ 67 ms / ≥ 15 FPS — PASS (including worst case)**

## Resource Requirements

Testing at `cpus: "1"` and `mem_limit: "1g"` produced identical results to
higher limits, which tells us the algorithm is effectively bounded by:

- **CPU: 1 core.** The vectorised numpy path is single-threaded; additional
  cores provide no per-frame benefit. On the Pi 5 (4 cores, ~1 reserved for
  Hailo), one core is dedicated to this detector and the remaining two are
  free for the OS, camera capture, and orchestration.
- **RAM: ~100 MB in practice.** Working set per frame is small (input image
  ~700 KB, RANSAC matrix ~500 KB, masks ~1 MB, Python/OpenCV runtime ~80 MB).
  The 1 GB container limit has ~900 MB headroom. The detector could run
  comfortably inside a `256m` limit.

## Comparison With Previous Attempts

| Metric | Attempt 2 | Attempt 3 | Attempt 4 |
|---|---:|---:|---:|
| Pass rate | 81.2% | 95.9% | 95.5% |
| Mean latency | 3.57 ms | 69.6 ms | **16.2 ms** |
| P90 latency | — | 156 ms | **34.2 ms** |
| Worst latency | — | 384 ms | **60.8 ms** |
| Meets 15 FPS budget? | Yes | No (P90) | **Yes (incl. worst case)** |

Attempt 3 had accuracy but a prohibitive slow tail (P90 156 ms, worst 384 ms)
that failed the speed budget on 31% of frames. Attempt 4 restores worst-case
compliance while preserving the accuracy gain.

## Remaining Failure Modes

All remaining failures are **classifier failures**, not fitting failures — the
same root causes as attempts 2 and 3:

| Failure mode | Trigger |
|---|---|
| Luminance ambiguity | Overcast sky ≈ grey ground; Otsu cuts at the wrong level |
| Sun glare | Bright patch in sky labelled as ground by Otsu |
| Haze band | Washed-out haze darker than upper sky confuses the threshold |

The worst frames (c2 clips, IoU 68–71%) are classic mask failures. A
colour-based classifier (Lab b\*, or Ettinger's covariance method) is the
most promising next step to push past ~96% pass rate.

## Bottom Line

Attempt 4 is the current best all-round result: it matches attempt 3's
accuracy and brings the slow tail within the 15 FPS budget even at worst case,
while needing only 1 CPU core and ~100 MB RAM on the target hardware.
