
## Train/Test Evaluation

seed=42 | train=86 | test=24

| Metric | Train | Test |
|---|---|---|
| N evaluated | 86 | 24 |
| N failed | 0 | 0 |
| FPS (excl. 10-frame warmup) | 490.1 | 492.4 |
| Mean latency (ms) | 2.04 | 2.03 |
| Mean angle error (°) | 11.23 | 7.68 |
| P90 angle error (°) | 28.44 | 14.79 |
| Mean position error (%H) | 13.85 | 19.44 |
| P90 position error (%H) | 36.56 | 54.24 |
| Mean IoU | N/A | N/A |
| Pass rate (Δθ<5° & Δρ<5%H) | 39.5% | 41.7% |
| mAP (threshold sweep) | 0.4201 | 0.4479 |

**mAP threshold breakdown:**

| Δθ max | Δρ/H max | Train precision | Test precision |
|---|---|---|---|
| 1° | 1% | 0.163 | 0.167 |
| 2° | 2% | 0.279 | 0.333 |
| 3° | 3% | 0.337 | 0.375 |
| 5° | 5% | 0.395 | 0.417 |
| 7° | 7% | 0.453 | 0.417 |
| 10° | 10% | 0.500 | 0.542 |
| 15° | 15% | 0.570 | 0.625 |
| 20° | 20% | 0.663 | 0.708 |

