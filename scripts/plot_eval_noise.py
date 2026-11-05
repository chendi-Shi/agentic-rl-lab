"""Draw the evaluation-noise figure from the real traces.

Left panel: the same model, same task set, five sampling seeds -- the spread is as
large as the effect being claimed, which is why single-run comparisons are
unreadable.

Right panel: each training run's paired improvement with its 95% interval, plus the
pooled estimate. Pairing on (task, seed) is what shrinks the error bars enough for
the effect to be visible.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRACES = Path("tmp_extract/handoff/traces")
OUT = Path("agentic-rl-lab/docs/figures/eval_noise.png")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    base_rows = load(TRACES / "q2_base.jsonl")
    base_cells = {(r["task_id"], r["seed"]): r["reward"]["r_task"] for r in base_rows}

    by_seed = {}
    for row in base_rows:
        by_seed.setdefault(row["seed"], []).append(row["reward"]["r_task"])
    seed_means = {s: st.mean(v) for s, v in sorted(by_seed.items())}

    runs = []
    pooled = []
    for path in sorted(TRACES.glob("ev_c*.jsonl")):
        cells = {(r["task_id"], r["seed"]): r["reward"]["r_task"] for r in load(path)}
        keys = sorted(set(base_cells) & set(cells))
        diffs = [cells[k] - base_cells[k] for k in keys]
        se = (st.stdev(diffs) / len(diffs) ** 0.5) if len(diffs) > 1 else 0.0
        runs.append((path.stem.replace("ev_", ""), st.mean(diffs), se))
        pooled.extend(diffs)

    p_mean = st.mean(pooled)
    p_se = st.stdev(pooled) / len(pooled) ** 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})

    seeds = list(seed_means)
    values = [seed_means[s] for s in seeds]
    bars = ax1.bar([str(s) for s in seeds], values, color="#4C78A8")
    overall = st.mean(base_cells.values())
    ax1.axhline(overall, color="#E45756", linestyle="--", linewidth=1.6,
                label=f"mean over all seeds = {overall:.3f}")
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
                 ha="center", fontsize=8)
    ax1.set_ylim(0, max(values) * 1.25)
    ax1.set_xlabel("sampling seed")
    ax1.set_ylabel("mean r_task")
    ax1.set_title("Same model, same 40 tasks, five seeds", fontsize=11)
    ax1.legend(fontsize=8, loc="lower right")
    spread = max(values) - min(values)
    ax1.text(0.02, 0.95, f"spread = {spread:.3f}", transform=ax1.transAxes,
             fontsize=9, va="top")

    labels = [r[0] for r in runs] + ["pooled"]
    means = [r[1] for r in runs] + [p_mean]
    ses = [r[2] for r in runs] + [p_se]
    colors = ["#54A24B"] * len(runs) + ["#E45756"]

    ypos = list(range(len(labels)))[::-1]
    for y, mean, se, color in zip(ypos, means, ses, colors):
        ax2.errorbar(mean, y, xerr=1.96 * se, fmt="o", color=color, capsize=4, markersize=6)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_yticks(ypos[::-1])
    ax2.set_yticklabels(labels[::-1], fontsize=9)
    ax2.set_xlabel("paired improvement over baseline (r_task), 95% CI")
    ax2.set_title("Pairing on (task, seed) makes the effect visible", fontsize=11)

    ax2.text(0.02, 0.02,
             f"unpaired SE ≈ {st.stdev([st.mean(v) for v in by_seed.values()]) / len(by_seed) ** 0.5:.3f}\n"
             f"paired SE (pooled) ≈ {p_se:.3f}",
             transform=ax2.transAxes, fontsize=8, va="bottom",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")

    print("\nper-seed base means:")
    for s, v in seed_means.items():
        print(f"  seed {s}: {v:.4f}")
    print(f"  spread: {spread:.4f}")
    print("\npaired per run:")
    for label, mean, se in runs:
        print(f"  {label}: {mean:+.4f} ± {1.96*se:.4f}")
    print(f"  pooled: {p_mean:+.4f} ± {1.96*p_se:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
