"""
tools/render_charts.py — render presentation bar charts as PNGs.

Pure matplotlib so the chart files can be embedded directly into the deck and
re-rendered deterministically by anyone who clones the repo.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ATTEMPTS_DIR = REPO_ROOT / "attempts"
OUT_DIR = REPO_ROOT / "presentation" / "charts"


def collect_attempt_results():
    rows = []
    for attempt_dir in sorted(ATTEMPTS_DIR.iterdir()):
        if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt-"):
            continue
        n = int(attempt_dir.name.split("-")[1])
        row = {"n": n, "name": attempt_dir.name}
        for label, ds in (("uav", "horizon_uav_dataset"), ("fpv", "video_clips_fpv_atv")):
            p = attempt_dir / f"full-eval-results-{ds}.json"
            if p.exists():
                s = json.loads(p.read_text())["summary"]
                row[f"{label}_pass"] = s["accuracy"]["pass_rate"] * 100
                row[f"{label}_lat"] = s["speed"]["latency_ms"]["mean"]
        rows.append(row)
    rows.sort(key=lambda r: r["n"])
    return rows


# Hand-curated short labels — the directory descriptors are too long and run
# into each other on the x-axis. These keep each attempt to <= 14 chars so
# rotated labels don't overlap.
_SHORT_LABELS = {
    1: "Otsu + scan",
    2: "Rotation-inv",
    3: "Top-N RANSAC",
    4: "Top-N tuned",
    5: "Efficient RS",
    6: "Dual channel",
    7: "Multi-cue",
    8: "+ Temporal",
    9: "+ DP boundary",
    10: "+ Sky envelope",
}


def short_label(n: int, name: str) -> str:
    return f"#{n}  {_SHORT_LABELS.get(n, name)}"


def render_dataset_chart(rows, dataset_key: str, title: str, out_path: Path) -> None:
    rows_with_data = [r for r in rows if f"{dataset_key}_pass" in r]
    labels = [short_label(r["n"], r["name"]) for r in rows_with_data]
    pass_rates = [r[f"{dataset_key}_pass"] for r in rows_with_data]

    fig, ax = plt.subplots(figsize=(11, 6.0))
    colors = []
    for r in rows_with_data:
        if r["n"] == 8:
            colors.append("#d62728")  # featured attempt — red highlight
        elif r["n"] == 10:
            colors.append("#2ca02c")  # current best — green
        else:
            colors.append("#4c72b0")
    bars = ax.bar(labels, pass_rates, color=colors)

    ax.set_ylim(0, 105)
    ax.set_ylabel("Pass rate  (%)", fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.axhline(60, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(95, color="#888888", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.grid(axis="y", alpha=0.25)

    for bar, rate in zip(bars, pass_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{rate:.1f}",
            ha="center", va="bottom", fontsize=10,
        )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_attempt_results()
    render_dataset_chart(
        rows, "uav",
        "Horizon-UAV (490 frames) — pass rate per attempt",
        OUT_DIR / "uav_progression.png",
    )
    render_dataset_chart(
        rows, "fpv",
        "FPV / ATV (120 frames) — pass rate per attempt",
        OUT_DIR / "fpv_progression.png",
    )


if __name__ == "__main__":
    main()
