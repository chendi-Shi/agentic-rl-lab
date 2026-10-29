"""Decide which tasks survive into an RL batch.

GRPO needs reward variance inside each group of samples for the same task. A task
that is always solved (or never solved) contributes nothing but compute.

    python scripts/difficulty_report.py --demo
    python scripts/difficulty_report.py --json traces/rewards.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.diagnostics import (  # noqa: E402
    classify_difficulty,
    detect_advantage_collapse,
    group_stats,
)

DEMO = {
    "always-solved": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "always-fails": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "learnable-a": [0.0, 1.0, 1.0, 0.0, 1.0, 0.5],
    "learnable-b": [0.0, 0.5, 1.0, 1.0, 0.0, 0.5],
    "borderline": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


def load(path: Path) -> dict[str, list[float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("expected a JSON object: {task_id: [rewards...]}")
    return {str(k): [float(x) for x in v] for k, v in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="path to {task_id: [rewards]} JSON")
    parser.add_argument("--demo", action="store_true", help="run on built-in sample data")
    parser.add_argument("--lo", type=float, default=0.2)
    parser.add_argument("--hi", type=float, default=0.8)
    args = parser.parse_args()

    rewards = DEMO if args.demo or not args.json else load(args.json)

    print(f"{'task':<18}{'n':>4}{'mean':>9}{'std':>9}{'pass':>8}  verdict")
    print("-" * 62)
    report = classify_difficulty(rewards, lo=args.lo, hi=args.hi)
    verdicts = (
        {t: "KEEP" for t in report.kept}
        | {t: "drop: too easy" for t in report.too_easy}
        | {t: "drop: too hard" for t in report.too_hard}
    )
    for stat in sorted(report.stats, key=lambda s: s.pass_rate):
        print(
            f"{stat.task_id:<18}{stat.n:>4}{stat.mean:>9.3f}{stat.std:>9.3f}"
            f"{stat.pass_rate:>8.0%}  {verdicts.get(stat.task_id, '-')}"
        )

    print()
    collapse = detect_advantage_collapse(rewards)
    print(collapse.summary())
    if not collapse.healthy:
        print("  Fix before training: resample or drop collapsed tasks, do not tune the LR.")
    print(f"  keep {len(report.kept)} / {len(report.stats)} tasks "
          f"({args.lo:.0%}-{args.hi:.0%} pass band)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
