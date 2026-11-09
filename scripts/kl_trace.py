"""Print the KL trajectory of a training run, which is the point of the mask ablation.

The coarse-mask run (reports/grpo-c1.json and friends) shows KL going 0.007 -> 10
over 200 steps while solved-per-group falls from 6.70 to 5.65. If the corrected
assistant mask is the cause, the same measurement on a proper-mask run should stay
flat. That single number decides whether the earlier degradation was a mask bug or
just a learning rate that was too high.

Usage:
    python scripts/kl_trace.py reports/grpo-m1.json [reports/grpo-c1.json ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path, buckets: int = 6) -> None:
    if not path.exists():
        print(f"{path.name}: missing")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    steps = payload.get("steps", [])
    if not steps:
        print(f"{path.name}: no steps")
        return

    kls = [s["kl"] for s in steps]
    rewards = [s["mean_reward"] for s in steps]
    solved = [s["solved"] for s in steps]
    graded = sum(1 for s in steps if s["reward_std"] > 1e-6)

    print(f"\n{path.name}  steps={len(steps)}  gradient_steps={graded} "
          f"({graded / len(steps):.0%})")
    print(f"{'range':<12}{'kl':>10}{'reward':>10}{'solved':>9}")
    width = max(1, len(steps) // buckets)
    for start in range(0, len(steps), width):
        chunk = slice(start, start + width)
        label = f"{start}-{min(start + width, len(steps)) - 1}"
        print(f"{label:<12}"
              f"{sum(kls[chunk]) / len(kls[chunk]):>10.4f}"
              f"{sum(rewards[chunk]) / len(rewards[chunk]):>10.3f}"
              f"{sum(solved[chunk]) / len(solved[chunk]):>9.2f}")

    first, last = kls[:10], kls[-10:]
    print(f"  kl first10={sum(first) / len(first):.4f}  "
          f"last10={sum(last) / len(last):.4f}  max={max(kls):.4f}")
    verdict = "STABLE" if max(kls) < 0.5 else "DIVERGING"
    print(f"  verdict: {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.reports:
        summarize(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
