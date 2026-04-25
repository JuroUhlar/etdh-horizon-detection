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

| Attempt | Method | Pass rate | Mean latency (dev) | Status |
|---|---|---|---|---|
| 1 — `attempts/attempt-1-otsu-column-scan/` | Grayscale → Otsu threshold → morph close → column-scan → `np.polyfit` | 62.4 % | 0.76 ms | Done, scored |
| 2 — `attempts/attempt-2-rotation-invariant/` | Same classifier; replaces column-scan with morph-gradient boundary + `cv2.fitLine` (Huber) | 81.2 % | 3.70 ms | Done, scored |
| 3 — `attempts/attempt-3-top-n-ransac/` | Same classifier; RANSAC multi-hypothesis + Huber refit | 95.5 % | 71.5 ms | Done, scored |
| 4 — `attempts/attempt-4-top-n-ransac_tuned/` | Same as 3; vectorised RANSAC scoring, boundary subsampling, fewer iterations | 95.5 % | 18.0 ms | Done, scored |

## Known failure modes & root causes

All remaining hard failures after attempt 2 are **classifier failures**, not fitter failures. Sky-mask IoU on worst cases is 0.7–0.9 — the classifier got most pixels right but the boundary it produced does not correspond to the real horizon.

| Failure mode | Trigger | Affected attempts |
|---|---|---|
| Luminance ambiguity | Overcast sky (dull grey) ≈ ground (dark greens) — Otsu splits along the wrong feature | 1, 2, 3, 4 |
| Sun glare | Bright glare patch on one side of the sky; Otsu labels glare = sky, real sky = ground | 1, 2, 3, 4 |
| Haze band | Washed-out haze above the real horizon is darker than upper sky; Otsu's cut lands at the wrong level | 1, 2 (partially), 3, 4 |

Rotation failures from attempt 1 (column-scan assumption) were resolved in attempt 2.

## Next-step options (ranked)

1. **Colour-based classifier** — split on Lab b* channel (sky skews blue, ground skews yellow/green). Small code delta on top of attempt 2/3's boundary extractor. Expected to fix luminance-ambiguity and glare cases.
2. **Ettinger's covariance method** — for each candidate line, score how well it separates pixels into two colour-coherent Gaussian groups. Classical UAV baseline; robust but ~80 extra lines. Suitable if colour thresholding alone doesn't reach the pass rate target.
3. **Benchmark on Raspberry Pi 5** — dev-machine timings (M-series Mac) are meaningless for the ≥15 FPS budget. Need real numbers before committing to a heavier classifier.
4. **Hough-transform variant** — extract edge segments, filter by length/slope, group collinear segments. An alternative route if colour cues are insufficient.

Background research on all of these methods (literature references, dataset survey, test-design notes): [`docs/research-horizon-detection.md`](docs/research-horizon-detection.md).
Known reference implementations: [`docs/inspiration-implementations.md`](docs/inspiration-implementations.md).

## Evaluation environment

Timing is validated inside a Docker container that approximates the Raspberry Pi 5 resource budget (3 CPU cores, 3.5 GB RAM) on any developer machine. See [`docs/evaluation-environment.md`](docs/evaluation-environment.md) for the full design rationale, what is and isn't simulated, and open items before the Dockerfile is written.

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
│   └── attempt-4-top-n-ransac_tuned/
│       ├── horizon_detect.py
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
- **Scores are always run** via `tools/evaluate.py <attempt-dir>` against the full 490-image Horizon-UAV dataset before an attempt is considered done; use `--dataset data/video_clips_ukraine_atv` for the secondary labelled set. Outputs are `full-eval-results-<dataset_dir_name>.json` per run.
- **Line representation internally:** `(vx, vy, x0, y0)` from `cv2.fitLine` — valid at all orientations. Convert to `(angle_deg, intercept_y)` for human output; convert to Hesse `(θ, ρ)` for metric computation.
- **Python environment:** `.venv/` at repo root.
