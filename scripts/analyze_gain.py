"""Where does the +0.033 actually come from?

The pooled result says training helps. It does not say how, and "which tasks
improved" is the first question anyone will ask. Three decompositions, all computed
from the traces:

1. by task family -- did the hard logic tasks improve, or only the easy ones?
2. by reward component -- is it answers, or just formatting and fewer invalid calls?
3. by outcome transition -- how many cells flipped fail->solve versus solve->fail?

Trace layout expected under --traces: ``q2_base.jsonl`` for the untrained model and
``ev_c*.jsonl`` for each training run, over the same (task, seed) cells.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def family(task_id: str) -> str:
    if task_id.startswith("agg-"):
        return "agg (6 orders)"
    for prefix in ("sub-max", "sub-min", "sub-count", "sub-over", "sub-median"):
        if task_id.startswith(prefix):
            return prefix.replace("sub-", "")
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=Path("tmp_extract/handoff/traces"))
    args = parser.parse_args()
    traces = args.traces

    base = {(r["task_id"], r["seed"]): r for r in load(traces / "q2_base.jsonl")}
    run_paths = sorted(traces.glob("ev_c*.jsonl"))
    runs = [load(path) for path in run_paths]
    if not base or not runs:
        raise SystemExit(f"missing traces under {traces}")

    # Every run evaluated the same (task, seed) cells, so per-cell values are the
    # mean across runs. Keying a dict by (task, seed) alone would silently keep only
    # the last run -- that bug made the first version of this analysis report a
    # negative result.
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rows in runs:
        for row in rows:
            by_cell[(row["task_id"], row["seed"])].append(row)
    cells = sorted(set(base) & set(by_cell))
    print(f"paired cells: {len(cells)}   runs per cell: {st.mean(len(v) for v in by_cell.values()):.1f}")

    def trained_mean(cell: tuple[str, int], name: str) -> float:
        return st.mean(r["reward"].get(name, 0.0) for r in by_cell[cell])

    print("\n## 1. by task family\n")
    fam_base: dict[str, list[float]] = defaultdict(list)
    fam_train: dict[str, list[float]] = defaultdict(list)
    for cell in cells:
        fam = family(cell[0])
        fam_base[fam].append(base[cell]["reward"]["r_task"])
        fam_train[fam].append(trained_mean(cell, "r_task"))
    print(f"{'family':<16}{'cells':>7}{'base':>8}{'trained':>9}{'diff':>8}")
    for fam in sorted(fam_base):
        b, t = st.mean(fam_base[fam]), st.mean(fam_train[fam])
        print(f"{fam:<16}{len(fam_base[fam]):>7}{b:>8.3f}{t:>9.3f}{t - b:>+8.3f}")

    print("\n## 2. by reward component\n")
    print(f"{'component':<14}{'base':>10}{'trained':>10}{'delta':>10}")
    for name in ("r_task", "r_format", "p_invalid", "p_steps", "p_error", "p_truncation"):
        b = st.mean(base[c]["reward"].get(name, 0.0) for c in cells)
        t = st.mean(trained_mean(c, name) for c in cells)
        print(f"{name:<14}{b:>10.4f}{t:>10.4f}{t - b:>+10.4f}")
    for name in ("n_steps", "n_calls"):
        b = st.mean(base[c]["reward"].get(name, 0.0) for c in cells)
        t = st.mean(trained_mean(c, name) for c in cells)
        print(f"{name:<14}{b:>10.2f}{t:>10.2f}{t - b:>+10.2f}")

    print("\n## 3. outcome transitions (per run, per cell)\n")
    # Counted per run: a cell value is a mean over nine runs, so thresholding that
    # mean at 0.999 counts "8 of 9 solved" as a failure and inverts the table.
    transitions: dict[tuple[str, str], int] = defaultdict(int)
    for rows in runs:
        for row in rows:
            cell = (row["task_id"], row["seed"])
            if cell not in base:
                continue
            before = base[cell]["reward"]["r_task"] >= 0.999
            after = row["reward"]["r_task"] >= 0.999
            transitions[("solve" if before else "fail", "solve" if after else "fail")] += 1
    total = sum(transitions.values())
    for (before, after), count in sorted(transitions.items(), key=lambda kv: -kv[1]):
        print(f"  {before:>5} -> {after:<5} {count:>5}  ({count / total:.4%})".replace("0000%", ".1%"))
    gained = transitions[("fail", "solve")]
    lost = transitions[("solve", "fail")]
    print(f"\n  over {total} (run, cell) pairs: +{gained} gained, -{lost} lost, net {gained - lost:+d}")

    print("\n## 4. largest per-task deltas\n")
    per_task: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for cell in cells:
        entry = per_task[cell[0]]
        entry[0] += base[cell]["reward"]["r_task"]
        entry[1] += trained_mean(cell, "r_task")
        entry[2] += 1
    ranked = sorted(
        ((tid, (v[1] - v[0]) / v[2]) for tid, v in per_task.items()), key=lambda kv: kv[1]
    )
    for task_id, delta in ranked[:3]:
        print(f"  {task_id:<30} {delta:+.3f}")
    print("  ...")
    for task_id, delta in ranked[-3:]:
        print(f"  {task_id:<30} {delta:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
