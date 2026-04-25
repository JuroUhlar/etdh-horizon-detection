# Attempt Comparison

This file compares the horizon-detection attempts in plain language. Numbers are from `tools/evaluate.py` (latest runs; attempt 3 is stochastic, so re-running without `--seed` can shift metrics slightly). Attempts 5 and 6 are not included here — they were short-lived experiments documented in their own `result.md` files; this comparison focuses on the attempts that actually moved the deployment story.

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
| Attempt 3 | Keep the same mask, but try many candidate lines, keep the ones with the most support, and refit the winner | Best line accuracy in the stack | Much slower; stochastic |
| Attempt 4 | Same pipeline as 3, but vectorised RANSAC scoring, subsampled boundary points, and fewer iterations | Very close to attempt 3 accuracy on Horizon-UAV, much faster | Same mask limits as 2/3; ATV accuracy can sit below 3 on some runs |
| Attempt 7 | Run RANSAC on **two** masks (grayscale + Lab b\*), pool the top hypotheses, then pick the winner by how cleanly it splits the frame into two color-coherent regions (Ettinger-style rerank with an angle prior). Filter out near-vertical boundary pixels before RANSAC, and abstain ("no_horizon") when the masks degenerate or no candidate is coherent enough. | First attempt that meaningfully helps on FPV/ATV without losing UAV; first attempt with a working `no_horizon` path | A handful of FPV ground-level treeline shots still fool it (Δθ up to ~38°); abstention thresholds are hand-picked, not swept |
| Attempt 8 | Keep attempt 7's candidate pool, but add a scene-change-gated temporal prior in the reranker and raise the coherence abstention floor | Best FPV/ATV pass rate so far; improves no-horizon abstention from 6/10 to 8/10 | Loses one UAV frame and increases FPV false abstentions; still does not solve the hardest treeline frames |

## Full Results

All numbers below are from:

```bash
.venv/bin/python tools/evaluate.py <attempt-dir>                                    # default: Horizon-UAV
.venv/bin/python tools/evaluate.py <attempt-dir> --dataset data/video_clips_fpv_atv
.venv/bin/python tools/evaluate.py <attempt-dir> --seed 0                          # pin stochastic detectors
```

The two datasets stress very different things, so we report them side by side rather than averaging — the takeaways are not the same.

### Horizon-UAV (`490` images, 480×480, every frame has a horizon)

| Metric | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 7 | Attempt 8 |
|---|---:|---:|---:|---:|---:|---:|
| Pass rate | 62.4% | 81.2% | 95.5% | 95.1% | **96.9%** | 96.7% |
| Mean Δθ | 10.461° | 7.313° | 1.091° | 1.078° | **0.994°** | 1.011° |
| P50 Δθ | 1.415° | 0.917° | 0.755° | 0.761° | **0.746°** | 0.755° |
| P90 Δθ | 36.793° | 32.036° | 2.331° | 2.296° | **2.021°** | 2.077° |
| Max Δθ | 85.059° | 88.744° | 7.613° | 7.704° | 8.201° | 8.201° |
| Mean Δρ (Hesse, px) | 70.744 | 36.700 | 10.201 | 10.625 | **8.836** | 8.947 |
| Mean Δρ / H | 0.147 | 0.076 | 0.021 | 0.022 | **0.018** | 0.019 |
| Mean Sky-mask IoU | 0.926 | 0.929 | 0.929 | 0.929 | 0.904 | 0.884 |
| Mean latency | 0.757 ms | 3.703 ms | 71.502 ms | 18.006 ms | **8.620 ms** | 9.2 ms |

Attempt 7 still wins on every Horizon-UAV line-fit metric. Attempt 8 gives up one frame because its higher coherence floor abstains more aggressively. The mean IoU drop (`0.929 → 0.904 → 0.884`) does not show up proportionally in the pass-gate metrics (Δθ, Δρ); it comes from frames where the *line* is on target but the emitted mask is offset or comes from the less IoU-friendly channel.

Attempt 3 (and, to a lesser extent, 4) is stochastic. If you run the commands above without `--seed`, metrics can wobble slightly; pass `--seed 0` to pin a reproducible result when you need to match a table exactly.

### FPV/ATV clips (`120` labelled frames, cropped + resized to ~625×480, 110 horizon + 10 no-horizon)

| Metric | Attempt 1 | Attempt 2 | Attempt 3 | Attempt 4 | Attempt 7 | Attempt 8 |
|---|---:|---:|---:|---:|---:|---:|
| Pass rate | 16.7% | 4.2% | 45.8% | 29.2% | 52.5% | **57.5%** |
| Mean Δθ (TP frames only) | 7.7° | 15.9° | 6.5° | 11.5° | 5.6° | **4.8°** |
| Mean Δρ (Hesse, px) | 59.7 | 100.0 | 59.8 | 102.2 | 52.8 | **48.9** |
| Mean Δρ / H | 0.124 | 0.208 | 0.124 | 0.213 | 0.110 | **0.102** |
| Mean latency | 0.79 ms | 29.8 ms | 731.6 ms | 58.9 ms | 29.3 ms | **26.5 ms** |
| Confusion matrix (TP / FN / FP / TN) | 110 / 0 / 10 / 0 | 110 / 0 / 10 / 0 | 110 / 0 / 10 / 0 | 110 / 0 / 10 / 0 | 105 / 5 / 4 / 6 | **100 / 10 / 2 / 8** |

A few things to read carefully here:

- **`Mean Δθ` is averaged over TP frames only** — frames where both the label and the detector say there's a horizon. The 10 no-horizon frames don't have a ground-truth line, so line errors are not defined for them.
- **Pass rate is over all 120 frames.** Attempts 1–4 emit a line on every frame (FP=10, TN=0), so the ceiling for them is 91.7% even if line accuracy on TP frames were perfect.
- **Attempt 7 is the first to abstain.** It correctly returns `no_horizon` on 6 of the 10 unlabelled frames (TN=6, FP=4) at the cost of 5 false abstentions on real-horizon frames (FN=5). Attempt 8 pushes this further to TN=8 / FP=2, but doubles false abstentions to FN=10.
- **Cropping the black side bars changes the result materially.** Attempt 3 goes from "catastrophically wrong" to a clear accuracy leader on ATV once the frame-border artefacts are removed.
- **Latency no longer scales cleanly with pixel count.** Attempt 1 becomes much cheaper after resizing, attempt 2 changes only modestly, and attempt 3 gets slower because the resized frames produce a denser boundary point cloud for its RANSAC stage.

## What Changed From Attempt To Attempt (Horizon-UAV)

### Attempt 1 -> Attempt 2

- Rotation handling improved a lot.
- Pass rate improved from `62.4%` to `81.2%`.
- Mean line-position error roughly halved.
- Mean latency increased from `0.757 ms` to `3.703 ms`, which is still very cheap on the dev machine.

Interpretation:

Attempt 2 shows that the first big problem was not only the mask. A lot of the error in attempt 1 came from the way the boundary was extracted and the way the line was fitted.

### Attempt 2 -> Attempt 3

- Pass rate improved again, from `81.2%` to `95.5%`.
- Mean angle error dropped sharply from `7.313°` to `1.091°`.
- Mean line-position error dropped from `36.700 px` to `10.201 px`.
- Mean latency jumped from `3.703 ms` to `71.502 ms`.

Interpretation:

Attempt 3 shows that the strongest remaining gains came from a better search over possible lines, not from a better sky mask. That is why IoU stays almost unchanged while the line metrics improve dramatically.

### Attempt 3 -> Attempt 4

- Pass rate is unchanged in this run: `95.5%` on Horizon-UAV.
- Mean angle error and mean `Δρ` stay in the same ballpark.
- Mean latency drops from `71.5 ms` to `18.0 ms` by vectorising the RANSAC scoring loop, subsampling boundary points, and running fewer iterations.

Interpretation: attempt 4 is the practical deployment variant of the same line-search idea when CPU time matters. On the FPV/ATV set, it is faster but does not always beat attempt 3 on every line metric (RANSAC randomness and slightly different per-frame work).

### Attempt 4 -> Attempt 7

- UAV pass rate ticks up from `95.1%` to `96.9%`. P90 angle error drops from `2.30°` to `2.02°`.
- FPV/ATV pass rate jumps from `29.2%` to `52.5%`. Mean angle error on TP frames roughly halves (`11.5°` -> `5.6°`).
- The detector can now say `no_horizon`. On FPV the confusion matrix moves from `110/0/10/0` to `105/5/4/6` — six previously-impossible TN, traded against four new FN.
- Mean latency on UAV goes from `18 ms` to `8.6 ms` on the dev host, but rises from `18.3 ms` to `22.3 ms` under the Docker / Pi 5 model (the Lab conversion and the per-channel RANSAC pass are the extra cost). Still well inside the 67 ms budget.

Interpretation:

The "more cues + smarter ranker + abstention" combination did three independent things:

- A second mask source (Lab b\*) gives RANSAC a second chance when the grayscale Otsu mask latches onto the wrong feature (luminance ambiguity, sun glare, haze band).
- The Ettinger-style rerank stops trusting inlier count as a proxy for "real horizon". Inlier count rewards whichever line happens to align with the strongest edge in the boundary mask; coherence on the original Lab pixels rewards lines that physically split the image into sky-like and ground-like regions. The angle prior keeps near-vertical tree-trunk hypotheses from winning by accident.
- The abstention path closes a long-standing accuracy ceiling on the FPV/ATV set: attempts 1–4 all hit a hard ~91.7% ceiling there because every no-horizon frame counted against them; attempt 7 is the first to climb past that ceiling.

What it did *not* fix: ground-level FPV shots through dense trees, where the *real* horizon is partially clipped by the canopy and the strongest color-coherent split lands along a treetop. Those are now the worst remaining failures (Δθ up to ~38° on the `fpv_treeline` frames).

### Attempt 7 -> Attempt 8

- Horizon-UAV pass rate drops slightly from `96.9%` to `96.7%`.
- FPV/ATV pass rate improves from `52.5%` to `57.5%`.
- FPV no-horizon abstention improves from `6 / 10` TN to `8 / 10` TN.
- FPV false abstentions increase from `5` to `10`, so the improvement is not free.
- Docker mean latency stays well inside budget: `21.5 ms` on Horizon-UAV and `26.5 ms` on FPV/ATV.

Interpretation:

Attempt 8 is not a new classifier. It is a video-stream-aware reranker: if the current frame looks continuous with the previous accepted frame, candidates close to the previous Hesse line get a soft score boost. The raw coherence floor is still enforced, and that floor was raised from attempt 7's `0.15` to `0.22`.

This helps on some continuous FPV frames where the candidate pool contains both a plausible horizon and a distracting region split. It does not help when the true boundary is absent from both Otsu-derived candidate pools. The worst `fpv_treeline` frames remain the same.

## What Changed On FPV/ATV After Cropping

The original ATV result was dominated by the dataset itself: large black side bars created strong artificial edges and pulled the detectors, especially the RANSAC pipeline, toward the frame border instead of the horizon. Rewriting the ATV frames to remove those bars changes the story completely.

All four attempts share the same first stage: an Otsu-style brightness threshold that splits the frame into "bright = sky" and "dark = ground", followed by a boundary-extraction step and a line fit. They differ only in how they extract the boundary and how they fit the line.

On Horizon-UAV that mask is mostly correct, because the upstream dataset is dominated by clear-sky aerial scenes where sky really is brighter than ground. On the FPV/ATV clip footage that assumption still breaks in many frames: treeline shots, road approaches, and ground-POV footage often have no clean brightness split. But after cropping the side bars, the boundary set is no longer polluted by a pair of huge vertical frame edges.

Once that first stage is wrong, each downstream choice amplifies the error differently:

- **Attempt 1** benefits immediately. Once the frame-border artefacts are gone, its simplistic column scan is much less likely to fit the border, so pass rate rises from `3.3%` to `16.7%`.
- **Attempt 2** does not benefit as much. It remains heavily constrained by the quality of the Otsu mask itself, so it is still often fitting the wrong boundary even though the most obvious artificial edges are gone.
- **Attempt 3** benefits the most in accuracy. Its aggressive search is no longer being handed an easy, dominant vertical border line, so it can often lock onto the real horizon. That is why pass rate jumps from `0.0%` to `45.0%` and mean `Δθ` on TP frames is far lower than on the uncropped dataset.
- **Attempt 4** is much faster on ATV than attempt 3, but line error on that set can be higher on average than attempt 3 in a given run; check `full-eval-results-video_clips_fpv_atv.json` when comparing.

In short: the more aggressive line search was not intrinsically wrong on ATV; it was being poisoned by bad input geometry. Once the dataset is cleaned up, RANSAC becomes useful again. The remaining failure mode is the brightness mask itself, plus the missing no-horizon path.

The 10 no-horizon frames are a separate failure orthogonal to all of this. None of the four attempts implements the `no_horizon` return path that the evaluator supports, so they take a hard 10/120 hit on classification before line accuracy is even measured.

## Recommended Reading Of The Results

- If you care most about **raw accuracy on the Horizon-UAV benchmark**, attempt 7 leads on both pass rate and the underlying line metrics; attempt 8 is one frame behind.
- If you care most about **speed**, attempt 1 is still the winner, but its accuracy is limited on both datasets.
- If you care about **best balance of simplicity, speed, and improvement on UAV**, attempt 2 is the most practical middle ground.
- If you care about **robustness across both datasets**, attempt 8 is the current best FPV/ATV option and has the best no-horizon abstention, while attempt 7 remains the best Horizon-UAV option. Attempts 1–4 share a brightness-mask first stage and a no-abstention policy that together cap their FPV/ATV ceiling at 91.7%.

## Docker / Pi 5 Performance

The latency numbers in the table above come from the development host (M-series Mac, unrestricted cores). To get numbers that are meaningful for the actual deployment target — a Raspberry Pi 5 running the horizon detector while a Hailo accelerator runs in parallel — we run the evaluator inside a Docker container constrained to approximate one available Pi core:

```bash
tools/bench_docker.sh   # 1 CPU core, 3.5 GB RAM, OMP_NUM_THREADS=1
```

Resource budget rationale: the Pi 5 has 4 Cortex-A76 cores; the Hailo-8L driver reserves ~1 core + ~512 MB. The container gets the remainder (3 cores, 3.5 GB) but with OMP pinned to 1 thread so the Python detector cannot silently multi-thread within a single frame. The x86 cores are ~1.5–2× faster per clock than Cortex-A76 NEON, so a passing result here is a necessary but not sufficient gate; final verification requires real hardware.

### Horizon-UAV results under Docker (1 CPU core, OMP_NUM_THREADS=1)

| Attempt | Pass rate | ms mean | ms p90 | FPS mean | FPS p90 | Speed gate |
|---|---:|---:|---:|---:|---:|:---:|
| Attempt 1 | 62.4% | 2.0 ms | 2.1 ms | 506 | 484 | ✓ PASS |
| Attempt 2 | 81.2% | 5.9 ms | 8.9 ms | 171 | 113 | ✓ PASS |
| Attempt 3 | 95.5% | 70.6 ms | 156.3 ms | 14.2 | 6.4 | ✗ FAIL |
| Attempt 4 | 95.1% | 17.6 ms | 33.9 ms | 56.7 | 29.5 | ✓ PASS |
| Attempt 7 | **96.9%** | 22.3 ms | 22.2 ms | 44.9 | 45.1 | ✓ PASS |
| Attempt 8 | 96.7% | 21.5 ms | 21.7 ms | 46.5 | 46.0 | ✓ PASS |

Speed gate: mean AND p90 latency ≤ 67 ms (15 FPS budget).

### What the Docker numbers tell us

Attempts 1 and 2 slow down noticeably under single-core Docker (2.2 ms vs 0.8 ms; 6.0 ms vs 3.7 ms) because OpenCV's morphological and gradient operations do have some internal threading that gets suppressed. Even so, both are far inside the budget.

Attempt 3 fails the speed gate: mean 70.7 ms is already above the 67 ms ceiling, and p90 jumps to 158 ms because hard RANSAC frames (dense boundary clouds) can take over 400 ms under a single constrained core. The accuracy gain over attempt 4 is negligible (both 95.x%) so attempt 3 is not a viable deployment candidate.

Attempt 4 sits in the practical sweet spot for the original RANSAC family: 95.1% pass rate, mean 17.6 ms, p90 33.9 ms — well within the 67 ms budget even at the 90th percentile. It is deterministically faster than attempt 3 because vectorised RANSAC scoring and boundary subsampling eliminate the worst-case per-frame explosion.

Attempt 7 spends roughly 4 ms more per frame on average than attempt 4 (Lab conversion + a second per-channel RANSAC pass + the rerank), but its p90 is actually *lower* than attempt 4's (22.2 ms vs 33.9 ms) because the orientation filter and boundary-point cap stop the worst-case point clouds from blowing up. It clears the 15 FPS gate by a wide margin and lifts pass rate to 96.9% on UAV (and 52.5% on FPV).

Attempt 8 adds only a tiny per-frame cost for a 16x16 scene thumbnail and a scalar temporal prior. Its Docker timing is effectively the same as attempt 7 on Horizon-UAV and a little faster on the final FPV run (`26.5 ms` vs attempt 7's `29.3 ms`), with both well under the 67 ms speed gate.

**Bottom line: attempts 4, 7, and 8 satisfy accuracy (≥ 95% on UAV) and speed (≤ 67 ms) inside the Pi 5 budget model. Attempt 7 remains best for Horizon-UAV. Attempt 8 is the better default if FPV/ATV robustness and no-horizon abstention matter more than one UAV frame. Attempt 3 still fails the speed gate under single-core constraints.**

## Practical Takeaway

The two datasets together tell a more honest story than either alone:

1. A naive brightness split plus simple line fit is not reliable enough on UAV, and is fundamentally wrong on FPV.
2. Cropping away large artificial borders can matter as much as algorithm changes; dataset hygiene was part of the ATV failure.
3. Making the boundary extraction and fit rotation-invariant solves a large class of UAV failures, but it still does not help much when the mask itself is not a horizon.
4. Searching across many candidate lines is the strongest improvement once the input geometry is sane, but it is still vulnerable when the mask is bad.
5. The two missing pieces — "stop trusting the brightness mask as a proxy for the horizon" and "let the detector abstain when no horizon is present" — were addressed in attempt 7 by pooling candidates from a second mask source, reranking on color-region coherence rather than inlier count, and abstaining when no candidate clears a coherence floor. Both UAV and FPV pass rates improved at the same time, which previous attempts had not managed.
6. Attempt 8 shows that video continuity can recover a few more FPV frames and reduce no-horizon false positives, but a single scalar coherence floor also creates more false abstentions.
7. The remaining failure mass is concentrated in FPV ground-level treeline shots (canopy partly clips the horizon) and in separating weak true horizons from true no-horizon frames. Those are the obvious targets for any future attempt.
