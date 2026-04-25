# hack-horizon — Project Scope & Engineering Guide

## Challenge brief

> "Implement real-time horizon detection onboard. Input: video stream from the camera.
> Output: horizon line parameters (angle, offset) or sky mask.
> Evaluation criteria: ≥15 FPS on RPi 5 while the detector is running simultaneously on
> Hailo. Accuracy of the horizon line (angle, offset)."
>
> Detection of ground targets without optical zoom is realistic at altitudes of up to
> 50–60 metres. At such altitudes, the sky enters the frame together with the ground —
> sometimes half the frame or more. For the task of searching for ground targets, the sky
> is pure overhead: the detector spends resources on it, and it is also a source of false
> positives (clouds, sun glare, birds). The obvious solution is to cut off everything
> above the horizon before feeding the image to the detector. The problem is that the
> horizon detection algorithm itself also consumes resources, and if it is heavier than
> the gain from cropping, there is no point. An algorithm must be found that, together
> with the detector, fits within the budget.

## Hard requirements

| # | Requirement | Detail |
|---|---|---|
| R1 | **Throughput** | ≥ 15 FPS on a Raspberry Pi 5 while a ground-target detector runs concurrently on the Hailo accelerator. The horizon detector must run on the Pi's CPU, not the Hailo. |
| R2 | **Output** | Horizon line parameters — angle (degrees, roll) and vertical offset (pixels or normalised) — and/or a binary sky mask. |
| R3 | **Accuracy** | The angle and offset must be accurate enough that sky pixels cropped by the horizon line do not cause the downstream detector to miss ground targets or regress on false-positive rate. No explicit numeric tolerance is stated in the brief; we use Δθ < 5° AND Δρ/H < 5% as our internal pass gate (derived from dataset evaluation). |
| R4 | **Onboard** | Algorithm runs on-device (no cloud call, no network round-trip). |

## Evaluation framework (internal)

Metrics are computed in Hesse normal form `(θ, ρ)` — finite at all orientations. Internal pass gate: **Δθ < 5° AND Δρ/H < 5%**.

Full metric definitions, aggregate statistics, and ground-truth conventions: [`docs/evaluation-metrics.md`](docs/evaluation-metrics.md).

**Primary benchmark dataset:** Horizon-UAV — 490 labelled 480×480 UAV frames with sky/land masks and `label.csv`. Details: [`data/horizon_uav_dataset/README.md`](data/horizon_uav_dataset/README.md). Evaluator: `tools/evaluate.py <attempt-dir>`.

## Attempt history

Pass rate and latency are from `tools/evaluate.py` on **Horizon-UAV** (490 frames) on the development host; see each attempt’s `full-eval-results-horizon_uav_dataset.json` for the exact run.

| Attempt | Method | UAV pass rate | FPV pass rate | Mean latency (dev host, UAV) | Mean latency (Docker / Pi 5 model, UAV) | Speed gate | Status |
|---|---|---|---|---|---|---|---|
| 1 — `attempts/attempt-1-otsu-column-scan/` | Grayscale → Otsu threshold → morph close → column-scan → `np.polyfit` | 62.4 % | — | 0.76 ms | 2.2 ms | ✓ PASS | Done, scored |
| 2 — `attempts/attempt-2-rotation-invariant/` | Same classifier; replaces column-scan with morph-gradient boundary + `cv2.fitLine` (Huber) | 81.2 % | — | 3.70 ms | 6.0 ms | ✓ PASS | Done, scored |
| 3 — `attempts/attempt-3-top-n-ransac/` | Same classifier; RANSAC multi-hypothesis + Huber refit | 95.5 % | — | 71.5 ms | 70.7 ms | ✗ FAIL | Done, scored |
| 4 — `attempts/attempt-4-top-n-ransac_tuned/` | Same as 3; vectorised RANSAC scoring, boundary subsampling, fewer iterations | 95.1 % | 29.2 % | 18.0 ms | 18.3 ms | ✓ PASS | Done, scored |
| 5 — `attempts/attempt-5-efficient-ransac/` | Tighter N=1 RANSAC: removed clustering, batched early stop, module-level RNG | 39.5 % | — | 2.04 ms | (not benched) | — | Done, scored — regression |
| 6 — `attempts/attempt-6-dual-channel-ransac/` | Dual-channel (gray + Lab b\*) Otsu, confidence-pick winner, 0° row-scan fallback | 92.5 % | 8.3 % | (not benched) | (not benched) | — | Done, scored — regression on FPV |
| 7 — `attempts/attempt-7-multicue-ettinger/` | Pool top-K from gray + Lab b\* RANSAC; Sobel orientation filter; rerank by Ettinger coherence × angle prior on 60×60 Lab thumbnail; abstain on degenerate masks / low coherence | **96.9 %** | **52.5 %** | 8.62 ms | 22.3 ms | ✓ PASS | Done, scored |

Docker environment: 1 CPU core, 3.5 GB RAM, `OMP_NUM_THREADS=1`. Speed gate: mean AND p90 latency ≤ 67 ms. Run via `tools/bench_docker.sh`.

Attempt 7 is the current best on both datasets and the first attempt with a working `no_horizon` abstention path (TN = 6 / 10 on the FPV labels).

## Known failure modes & root causes

All remaining hard failures after attempt 2 are **classifier failures**, not fitter failures. Sky-mask IoU on worst cases is 0.7–0.9 — the classifier got most pixels right but the boundary it produced does not correspond to the real horizon.

| Failure mode | Trigger | Affected attempts |
|---|---|---|
| Luminance ambiguity | Overcast sky (dull grey) ≈ ground (dark greens) — Otsu splits along the wrong feature | 1, 2, 3, 4 (partially fixed in 7 by Lab b\* fallback) |
| Sun glare | Bright glare patch on one side of the sky; Otsu labels glare = sky, real sky = ground | 1, 2, 3, 4 (partially fixed in 7 by Ettinger rerank) |
| Haze band | Washed-out haze above the real horizon is darker than upper sky; Otsu's cut lands at the wrong level | 1, 2 (partially), 3, 4 (partially fixed in 7) |
| FPV treeline / canopy | Ground-level FPV through dense trees: real horizon is partially clipped by canopy; the strongest color-coherent split lands along a treetop | All. Worst remaining failure mode in attempt 7 — Δθ up to 38° on `04_10m34s-11m00s_fpv_treeline_*` frames |
| No-horizon false positive | Sky-only / ground-only frames not quite degenerate enough to trip the abstention threshold; just enough texture coherence to clear the floor | 1–6 fit a line on every frame; attempt 7 abstains on 6 of 10 (4 FP remaining) |

Rotation failures from attempt 1 (column-scan assumption) were resolved in attempt 2.

## Next-step options (ranked)

1. **Better treeline behaviour** — currently the worst failure mass on FPV. Options: detect "canopy clipping" via row-wise variance profile and abstain; or fit two-line model (canopy line + horizon line) and pick the upper.
2. **Calibrate abstention thresholds with a sweep** — `_DEGENERATE_FRACTION` and `_FALLBACK_COHERENCE` in attempt 7 were hand-picked. A proper FN/FP sweep on FPV would likely move ~3 of the remaining 9 errors.
3. **Hough-transform variant** — extract edge segments, filter by length/slope, group collinear segments. Independent enough from the current pipeline that a third candidate source might help on cases where both Otsu masks fail.
4. **Real Raspberry Pi 5 benchmark** — the Docker model gates the speed budget honestly enough for development, but a real-Pi run is still required before declaring the requirement met.

Background research on all of these methods (literature references, dataset survey, test-design notes): [`docs/research-horizon-detection.md`](docs/research-horizon-detection.md).
Known reference implementations: [`docs/inspiration-implementations.md`](docs/inspiration-implementations.md).

## Evaluation environment

Timing is validated inside a Docker container that approximates the Raspberry Pi 5 resource budget (1 CPU core, 3.5 GB RAM, `OMP_NUM_THREADS=1`) on any developer machine. See [`docs/evaluation-environment.md`](docs/evaluation-environment.md) for the full design rationale, what is and isn't simulated, and open items before the Dockerfile is written.

## Repo layout

```
hack-horizon/
├── CLAUDE.md                        # this file — project scope and guide
├── attempts/
│   ├── attempt-1-otsu-column-scan/
│   │   ├── horizon_detect.py        # detector implementation
│   │   └── result.md                # method, scores, analysis
│   ├── attempt-2-rotation-invariant/
│   │   ├── horizon_detect.py
│   │   └── result.md
│   ├── attempt-3-top-n-ransac/
│   │   ├── horizon_detect.py
│   │   └── requirements.txt
│   ├── attempt-4-top-n-ransac_tuned/
│   │   ├── horizon_detect.py
│   │   └── result.md
│   ├── attempt-5-efficient-ransac/
│   │   ├── horizon_detect.py
│   │   └── result.md
│   ├── attempt-6-dual-channel-ransac/
│   │   ├── horizon_detect.py
│   │   └── result.md
│   └── attempt-7-multicue-ettinger/
│       ├── horizon_detect.py
│       ├── requirements.txt
│       └── result.md
├── data/
│   └── horizon_uav_dataset/         # 490-image benchmark (images/, masks/, label.csv)
├── docs/
│   ├── evaluation-metrics.md        # metric definitions and conventions
│   ├── research-horizon-detection.md
│   └── inspiration-implementations.md
└── tools/
    └── evaluate.py                  # batch evaluator — run against any attempt dir
```

## Conventions

- **Each attempt** lives in its own `attempts/attempt-N-<name>/` directory with a `horizon_detect.py` exposing a `detect_horizon(image_bgr) -> list[dict]` API and a `result.md` documenting method, scores, and failure analysis.
- **Scores are always run** via `tools/evaluate.py <attempt-dir>` against the full 490-image Horizon-UAV dataset before an attempt is considered done; use `--dataset data/video_clips_fpv_atv` for the secondary labelled set. Outputs are `full-eval-results-<dataset_dir_name>.json` per run.
- **Line representation internally:** `(vx, vy, x0, y0)` from `cv2.fitLine` — valid at all orientations. Convert to `(angle_deg, intercept_y)` for human output; convert to Hesse `(θ, ρ)` for metric computation.
- **Python environment:** `.venv/` at repo root.
