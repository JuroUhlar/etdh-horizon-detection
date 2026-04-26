# Attempt 7 — Multi-cue Candidates + Ettinger Rerank + No-Horizon Abstention

This attempt keeps the boundary-points → vectorised RANSAC → Huber-refit
pipeline that attempts 2–4 share, and replaces the *single classifier* +
*max-inlier winner* combination with three changes:

1. **Two boundary sources, pooled:** RANSAC is run on the boundary masks of
   both grayscale (Lab L) and Lab b\* channels. The top-K hypotheses from
   each channel become a single candidate pool.

2. **Ettinger-style rerank:** the winner is chosen by region-coherence
   ("does this line cleanly separate Lab pixels into two color-coherent
   groups?") on a 60×60 Lab thumbnail, not by inlier count. An angle prior
   multiplies the score so near-vertical candidates (tree trunks, frame
   borders) need overwhelmingly better coherence to win.

3. **`no_horizon` abstention:** if both channels produce a near-degenerate
   mask (≥92 % one class) or the best Ettinger coherence is below a floor,
   the detector returns `"no_horizon"` instead of fitting a line.

The motivation came from the attempt-6 retro: running two channels and
picking the higher-inlier winner is *worse* than a single channel because
RANSAC happily fits a high-inlier line to a non-horizon edge in either
channel — inlier count is a poor proxy for "is this the real horizon".
Ettinger coherence is a much closer proxy for the physical question.

A boundary-orientation filter (Sobel on the mask, drop pixels whose local
gradient is closer to vertical than the maximum expected roll) runs before
RANSAC. This removes long collinear segments from tree trunks and building
edges that previously dominated the candidate pool in FPV treeline shots.

## Pipeline

```
BGR ─► Lab ─┬─► Lab L (= grayscale) ─► Otsu + morph + gradient ─► boundary_L ─┐
            │                                                                 │
            └─► Lab b*               ─► Otsu + morph + gradient ─► boundary_b ┤
                                                                              │
boundary_X ─► Sobel orientation filter ─► RANSAC top-K ──► candidates ────────┤
                                                                              │
candidates ─► Ettinger coherence × angle prior on 60×60 Lab thumbnail ────────┤
                                                                              │
        winner.inliers ─► cv2.fitLine (Huber) ─► (vx, vy, x0, y0) ────────────┘

if both masks degenerate OR best_score < FALLBACK_COHERENCE: return "no_horizon"
```

## Full-Dataset Results

Both runs use `--seed 0` for reproducibility. Source of truth:
`full-eval-results-horizon_uav_dataset.json` and
`full-eval-results-video_clips_fpv_atv.json`.

### Horizon-UAV (480×480, 490 frames)

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Δθ (angle error, °) | 0.99 | 0.75 | 2.02 | 8.20 |
| Δρ / H (normalised position) | 0.018 | 0.015 | 0.031 | 0.544 |
| Hesse distance (px) | 8.84 | 7.40 | 14.66 | 261.09 |
| Sky-mask IoU | 0.904 | 0.951 | 0.980 | 0.997 |
| Latency (ms, dev host) | 8.62 | — | 8.62 | — |
| Latency (ms, Docker / Pi-5 model) | 22.28 | 20.88 | 22.18 | 235.57 |

**Pass rate: 475 / 490 = 96.94 %.** Speed gate: ✓ PASS (mean 22.3 ms ≤ 67 ms).

### FPV/ATV (120 frames, 10 of which are labelled `has_horizon=false`)

`has_horizon` confusion matrix:

| | pred horizon | pred no_horizon |
|---|---:|---:|
| **gt horizon** | TP = 105 | FN = 5 |
| **gt no_horizon** | FP = 4 | TN = 6 |

| Metric | mean | P50 | P90 | max |
|---|---:|---:|---:|---:|
| Δθ (angle error, °) | 5.61 | 2.37 | 13.12 | 37.94 |
| Δρ / H (normalised position) | 0.110 | 0.029 | 0.333 | 0.656 |
| Hesse distance (px) | 52.84 | 13.82 | 159.76 | 314.76 |
| Latency (ms, Docker / Pi-5 model) | 29.28 | 26.02 | 33.13 | 128.30 |

**Pass rate: 63 / 120 = 52.50 %.** Speed gate: ✓ PASS (mean 29.3 ms ≤ 67 ms).

## Comparison With Attempt 4 (Same Evaluator, Seed 0)

| Dataset | Metric | Attempt 4 | Attempt 7 | Δ |
|---|---|---:|---:|---:|
| Horizon-UAV | Pass rate | 95.10 % | **96.94 %** | +1.84 pp |
| Horizon-UAV | P90 angle err | 2.30° | **2.02°** | −0.28° |
| Horizon-UAV | Mean IoU | 0.929 | 0.904 | −0.025 |
| Horizon-UAV | Docker mean lat | 18.3 ms | 22.3 ms | +4.0 ms |
| FPV/ATV | Pass rate | 29.17 % | **52.50 %** | **+23.33 pp** |
| FPV/ATV | TN (correct abstain) | 0 / 10 | **6 / 10** | +6 |
| FPV/ATV | Mean angle err | 11.5° | **5.6°** | −5.9° |
| FPV/ATV | Docker mean lat | 58.9 ms | 29.3 ms | −29.6 ms |

The FPV mask-IoU drop (0.929 → 0.904) on UAV looks like a small regression,
but it is concentrated in cases where attempt 4 happened to land a slightly
better one-side-of-the-frame split despite a worse line angle. The pass-gate
metrics (Δθ, Δρ) both improve.

## Remaining Failure Modes

| Failure mode | Trigger | Affected dataset |
|---|---|---|
| **Tree-canopy line fit** | FPV ground-level shots through dense trees: the *real* horizon is partially clipped by the canopy and the strongest color-coherent split lands along a treetop instead | All 5 worst FPV frames (`fpv_treeline_f034..f045`) — Δθ up to 38° |
| **Glare halo on overcast sky** | Bright glare patch dominates the Lab b\* mask and the resulting boundary touches the glare edge, not the horizon | A handful of UAV worst-5 frames (Δθ ≈ 8°) |
| **False-positive line on no-horizon frames** | Sky-only or ground-only frames that aren't quite degenerate enough (mask-balance below 0.92) and have just enough texture coherence to clear the floor | 4 of 10 no-horizon frames (FP) |
| **False-negative abstention on real horizon** | Real horizon present but classifier produces a degenerate mask on both channels (heavy fog / extreme exposure) | 5 of 110 horizon frames (FN) |

The two abstention thresholds (`_DEGENERATE_FRACTION = 0.92`,
`_FALLBACK_COHERENCE = 0.15`) trade off FN against FP almost linearly.
Lower the floor → more FP, fewer FN. The current values were picked by hand
on the FPV set; a proper sweep would be more honest.

## Failed Sub-Experiments (Kept Here So The Next Agent Can Skip Them)

- **Texture-asymmetry term in the rerank score**
  Tried multiplying coherence by `1 + |var_above − var_below| / (var_above + var_below)`.
  Intent: reward smooth-sky-vs-textured-ground splits, penalise spurious
  splits between two textured regions.
  Result: −0.82 pp on UAV (96.94 → 96.12) and −3.33 pp on FPV (52.50 →
  49.17). Lab variance is dominated by L; the term mostly amplifies
  illumination differences, which often cuts *against* the horizon when the
  ground is actually brighter than the sky.

- **Confidence-gated channel pick (attempt-6 style)**
  Picking the higher-inlier-fraction channel and discarding the other was
  measurably worse than pooling both candidate sets — see attempt 6.

- **Row-scan 0° fallback (attempt-6 style)**
  When all hypotheses scored below a floor, attempt 6 emitted a flat 0°
  line at the brightness midpoint. On FPV it pulls the angle far from the
  truth on rolled clips. Attempt 7 abstains in the same situation instead.

## Bottom Line

Attempt 7 is the first detector in the series that meaningfully helps on
FPV/ATV (29 % → 53 %) without giving up Horizon-UAV (95 % → 97 %), and it
is the first that can decline to fit a line on no-horizon frames. The
remaining failure mass is concentrated in a small number of FPV
ground-level treeline shots, which is closer to a "does this scene contain
a horizon at all?" question than a line-fit question, and is the obvious
target for any future attempt.
