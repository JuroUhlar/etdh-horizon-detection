
## Train/Test Evaluation

seed=42 | train=388 | test=102

| Metric | Train | Test |
|---|---|---|
| N evaluated | 388 | 102 |
| N failed | 0 | 0 |
| FPS (excl. 10-frame warmup) | 276.2 | 277.9 |
| Mean latency (ms) | 3.62 | 3.60 |
| Mean angle error (°) | 1.33 | 1.05 |
| P90 angle error (°) | 2.55 | 2.20 |
| Mean position error (%H) | 2.37 | 2.34 |
| P90 position error (%H) | 4.14 | 4.38 |
| Mean IoU | 0.798 | 0.774 |
| Pass rate (Δθ<5° & Δρ<5%H) | 92.5% | 91.2% |
| mAP (threshold sweep) | 0.7848 | 0.7904 |

**mAP threshold breakdown:**

| Δθ max | Δρ/H max | Train precision | Test precision |
|---|---|---|---|
| 1° | 1% | 0.113 | 0.167 |
| 2° | 2% | 0.531 | 0.520 |
| 3° | 3% | 0.784 | 0.794 |
| 5° | 5% | 0.925 | 0.912 |
| 7° | 7% | 0.964 | 0.961 |
| 10° | 10% | 0.982 | 0.990 |
| 15° | 15% | 0.990 | 0.990 |
| 20° | 20% | 0.990 | 0.990 |

