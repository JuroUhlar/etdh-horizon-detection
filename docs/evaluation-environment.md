# Evaluation Environment — Docker Scope

## Purpose

Provide a reproducible, cross-platform execution environment that approximates the
resource budget of a Raspberry Pi 5 running the horizon detector, so that timing
measurements taken on any developer machine (Windows/WSL, macOS, Linux) are
comparable and conservative relative to the real target.

## Design decisions

### Architecture: x86 native with resource limits (not ARM emulation)

Running `--platform linux/arm64` under QEMU on an x86 host produces ~5–20× slowdown
that makes timing numbers useless. Instead we constrain the container to the Pi 5's
**CPU core count and RAM budget** so that throughput measurements are in the right
ballpark. This is a deliberate trade-off: instruction-set differences (AVX vs NEON) mean
the numbers are an approximation, not a ground truth. They are good enough for
go/no-go decisions during development; final FPS verification must be done on real
hardware.

### Hailo accelerator headroom

On the real device the Hailo-8L accelerator has its own compute die, but its Linux
driver and the host-side inference orchestration consume Pi CPU cycles — estimated
at roughly **1 core + 512 MB RAM** of continuous overhead. Rather than simulate a
dummy Hailo process inside the container, we simply **reduce the container's visible
budget** by that amount and document the assumption. This means:

| Resource | RPi 5 (4 GB) physical | Hailo driver reserve | Available pool | Container limit |
|---|---|---|---|---|
| CPU cores | 4 | ~1 core | 3 cores | **1 core** |
| RAM | 4 096 MB | ~512 MB (driver + OS) | ~3 584 MB | **3 584 MB (~3.5 GB)** |

The RAM limit mirrors the full remaining pool — 3.5 GB is far more than the algorithm needs (~MB range for 480×480 frames), so the container will never be OOM-limited in a way the Pi wouldn't also hit.

The CPU limit is deliberately set to **1 core**, not 3, for a second reason: the horizon detector runs in a single Python thread per frame, so the other two available cores only help if OpenCV parallelises internally (which `OMP_NUM_THREADS=1` suppresses). Setting `cpus: "1"` and all thread-count env vars to 1 forces fully serial execution, matching the worst-case single-core timing. On a Pi the detector *could* use those extra cores via BLAS/OMP threading — keeping them available would be more realistic, but it would make the container faster than the serial case, not slower. Choosing the stricter model means a passing container result is a stronger gate, not a weaker one.

If the horizon detector comfortably hits ≥15 FPS inside these limits it will have
sufficient headroom on the real device. If it just barely passes, treat that as a
warning flag and re-verify on hardware.

### Software stack

| Component | Version | Rationale |
|---|---|---|
| Base image | `python:3.14-slim-trixie` | Debian 13 Trixie — same OS generation as the latest RPi OS; slim keeps the image lean |
| Python | 3.14 | Latest stable; matches the target RPi OS Trixie stack |
| OpenCV | 4.10.x (PyPI `opencv-python-headless`) | Headless build — no GUI deps needed in a container; matches what we'd install on the Pi |
| NumPy | latest stable (≥2.0) | |
| No CUDA / no GPU | — | Pi 5 has no CUDA; container must not rely on GPU acceleration |

## What is NOT simulated

| Gap | Impact | Mitigation |
|---|---|---|
| ARM NEON SIMD vs x86 AVX | Some OpenCV kernels run at different speeds per ISA | Accept ±20% timing variance; use real hardware for final gate |
| Thermal throttling | Pi throttles under sustained load; container does not | Add a 10–15% safety margin when interpreting container FPS numbers |
| SD-card / USB I/O latency | Container uses host filesystem via bind-mount; Pi uses SD | Only matters if the algorithm streams frames from disk; keep dataset in RAM (`/dev/shm`) if IO becomes a bottleneck |
| Actual Hailo driver | Not running | Accounted for via the CPU/RAM reserve above |
| Camera capture overhead | Not measured | Evaluate pure algorithm latency; subtract from the 1/15 s frame budget manually |

## Configuration

Resource limits are hardcoded in `docker-compose.yml`. To run with an 8 GB variant
or different limits, edit those values directly.

## Per-experiment dependencies

Each attempt ships its own `requirements.txt`. There is no root-level requirements
file. Install before running:

```bash
docker compose run --rm horizon sh -c \
  "pip install -q -r attempts/<N>/requirements.txt && python attempts/<N>/horizon_detect.py ..."
```

## Directory layout

```
hack-horizon/
├── docker/
│   ├── Dockerfile           # base image only — no pip installs
│   └── .dockerignore
└── docker-compose.yml       # resource limits + bind-mount wiring
```

## Open items

- [ ] Confirm which script(s) the container needs to invoke as its primary use case
      (one-off image, batch evaluate, or live video stream). Currently no fixed
      entrypoint — any script can be passed to `docker compose run --rm horizon python ...`.
- [ ] Pin exact package versions in each attempt's `requirements.txt` once the image
      builds cleanly (for reproducibility across team members).

### Resolved

- **Bind-mount vs bake-in:** `data/` and all source are bind-mounted at runtime.
  The image only contains the Python runtime and apt-installed system libs.
- **Evaluator location:** `tools/evaluate.py` runs inside the container against the
  bind-mounted dataset and source.
- **Root-level requirements:** removed — dependencies stay scoped to each attempt.
