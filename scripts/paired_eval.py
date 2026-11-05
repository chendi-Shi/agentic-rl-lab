"""Paired analysis: base vs each trained adapters on identical (task, sample) cells.

The summary table compares means, which is weak here: the baseline was measured
once, and one 200-sample estimate of a 56% pass rate carries a standard error of
about 3.5 points -- the same size as the effect being claimed.

Both the baseline and every trained run evaluated the same 40 tasks with the same
five sampling seeds, so the cells can be paired instead: for cell (task, k) take
the base reward and the trained reward, subtract, and test the mean difference.
Pairing removes the between-task difficulty variation, which is most of the noise.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

BASE = Path("/root/agentic-rl-lab/traces/q2_base.jsonl")
TRAINED = sorted(Path("/root/agentic-rl-lab/traces").glob("ev_c*.jsonl"))


def load_cells(path: Path) -> dict[tuple[str, int], float]:
    """Map (task_id, sample index) -> r_task. Rows come in seed order per task."""

    cells: dict[tuple[str, int], float] = {}
    seen: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row["task_id"]
        index = seen.get(task, 0)
        seen[task] = index + 1
        cells[(task, index)] = float(row["reward"].get("r_task", 0.0))
    return cells


def paired_summary(base: dict, other: dict, label: str) -> dict:
    keys = sorted(set(base) & set(other))
    diffs = [other[k] - base[k] for k in keys]
    mean = st.mean(diffs)
    sd = st.stdev(diffs) if len(diffs) > 1 else 0.0
    se = sd / (len(diffs) ** 0.5)
    wins = sum(1 for d in diffs if d > 1e-9)
    losses = sum(1 for d in diffs if d < -1e-9)
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    print(
        f"{label:<8} n={len(diffs):>4}  base={st.mean(base[k] for k in keys):.4f}  "
        f"trained={st.mean(other[k] for k in keys):.4f}  "
        f"diff={mean:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  "
        f"wins={wins} losses={losses}"
    )
    return {"label": label, "n": len(diffs), "diff": mean, "lo": lo, "hi": hi,
            "wins": wins, "losses": losses}


def main() -> int:
    if not BASE.exists():
        print(f"missing {BASE}")
        return 1
    base = load_cells(BASE)
    print(f"base cells: {len(base)}")

    rows = []
    pooled: list[float] = []
    pooled_wins = pooled_losses = 0
    for path in TRAINED:
        cells = load_cells(path)
        row = paired_summary(base, cells, path.stem.replace("ev_", ""))
        rows.append(row)
        keys = sorted(set(base) & set(cells))
        pooled.extend(cells[k] - base[k] for k in keys)
        pooled_wins += row["wins"]
        pooled_losses += row["losses"]

    if pooled:
        mean = st.mean(pooled)
        sd = st.stdev(pooled)
        se = sd / (len(pooled) ** 0.5)
        lo, hi = mean - 1.96 * se, mean + 1.96 * se
        print(
            f"\nPOOLED across {len(TRAINED)} runs: n={len(pooled)} "
            f"diff={mean:+.4f}  CI=[{lo:+.4f}, {hi:+.4f}]  "
            f"wins={pooled_wins} losses={pooled_losses}"
        )
        # Sign test on the per-run means.
        above = sum(1 for r in rows if r["diff"] > 0)
        print(f"runs above baseline: {above}/{len(rows)}")
        Path("/root/paired.json").write_text(
            json.dumps({"rows": rows, "pooled": {"diff": mean, "lo": lo, "hi": hi,
                                                 "n": len(pooled),
                                                 "wins": pooled_wins,
                                                 "losses": pooled_losses}}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
