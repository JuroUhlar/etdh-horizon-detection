#!/usr/bin/env bash
# tools/bench_docker.sh — run all attempts inside Docker and print a comparison table.
#
# Usage:
#   tools/bench_docker.sh                     # full 490-frame eval, all attempts
#   tools/bench_docker.sh --limit 50          # quick smoke-test
#   tools/bench_docker.sh --dataset data/video_clips_fpv_atv
#
# All extra arguments are forwarded to tools/evaluate.py for every attempt.
# Results are also written to each attempt's full-eval-results-<dataset>.json.
#
# Resource budget (docker-compose.yml): 1 CPU core, 3.5 GB RAM, OMP_NUM_THREADS=1.
# See docs/evaluation-environment.md for the rationale.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ATTEMPTS=(
  attempts/attempt-1-otsu-column-scan
  attempts/attempt-2-rotation-invariant
  attempts/attempt-3-top-n-ransac
  attempts/attempt-4-top-n-ransac_tuned
)

EVAL_ARGS=("$@")

# ---------------------------------------------------------------------------
# Build image once
# ---------------------------------------------------------------------------
echo "Building Docker image..."
docker compose build --quiet
echo ""

# ---------------------------------------------------------------------------
# Run each attempt
# ---------------------------------------------------------------------------
for attempt in "${ATTEMPTS[@]}"; do
  name="$(basename "$attempt")"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  $name"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  docker compose run --rm horizon sh -c \
    "pip install -q --root-user-action=ignore -r $attempt/requirements.txt && \
     python tools/evaluate.py $attempt ${EVAL_ARGS[*]+"${EVAL_ARGS[*]}"}"
  echo ""
done

# ---------------------------------------------------------------------------
# Summary table (parsed from JSON result files on the host)
# ---------------------------------------------------------------------------
python3 - <<'PYEOF'
import json, pathlib, sys

attempts = [
    "attempt-1-otsu-column-scan",
    "attempt-2-rotation-invariant",
    "attempt-3-top-n-ransac",
    "attempt-4-top-n-ransac_tuned",
]

PASS_FPS = 15.0

rows = []
for name in attempts:
    pattern = f"attempts/{name}/full-eval-results-*.json"
    paths = sorted(pathlib.Path(".").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not paths:
        rows.append((name, None))
        continue
    with open(paths[-1]) as f:
        data = json.load(f)
    s = data.get("summary", {})
    rows.append((name, s))

if not any(r[1] for r in rows):
    sys.exit(0)

W = 44
print("━" * (W + 52))
print(f"{'ATTEMPT':<{W}}  {'PASS%':>6}  {'ms mean':>7}  {'ms p90':>7}  {'FPS mean':>9}  {'FPS p90':>8}  {'SPEED':>6}")
print("━" * (W + 52))

for name, s in rows:
    if s is None:
        print(f"  {name:<{W-2}}  {'(no results)':>46}")
        continue
    acc   = s.get("accuracy", {})
    speed = s.get("speed", {})
    lat   = speed.get("latency_ms") or {}
    fps   = speed.get("fps") or {}
    pr    = acc.get("pass_rate", 0) * 100
    ms_m  = lat.get("mean", 0)
    ms_p  = lat.get("p90", 0)
    fp_m  = fps.get("mean", 0)
    fp_p  = fps.get("p90", 0)
    verd  = speed.get("verdict", "?")
    mark  = "✓" if verd == "PASS" else ("~" if verd == "WARN" else "✗")
    print(f"  {name:<{W-2}}  {pr:>5.1f}%  {ms_m:>7.1f}  {ms_p:>7.1f}  {fp_m:>9.1f}  {fp_p:>8.1f}  {mark} {verd}")

print("━" * (W + 52))
print(f"  Speed gate: mean AND p90 latency ≤ {1000/PASS_FPS:.0f} ms  ({PASS_FPS:.0f} FPS)")
print(f"  Environment: 1 CPU core, OMP_NUM_THREADS=1  (conservative Pi 5 single-core model)")
PYEOF
