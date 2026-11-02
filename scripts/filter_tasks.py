"""Drop tasks that cannot teach GRPO anything.

    python scripts/filter_tasks.py \
        --tasks tasks/stage1-train.json \
        --store traces/base_sft.jsonl \
        --out tasks/stage1-kept.json

Keeps tasks whose pass rate lands in [--lo, --hi]. Run this after the baseline
and before training: a set that is all-solved or all-failed trains nothing while
still billing by the hour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.store import TraceStore  # noqa: E402
from arlab.tasks.stage_io import dump_tasks, load_tasks  # noqa: E402
from arlab.train.filtering import filter_tasks, pass_rates  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config-id", default=None, help="only use rollouts from this config")
    parser.add_argument("--lo", type=float, default=0.2)
    parser.add_argument("--hi", type=float, default=0.8)
    parser.add_argument(
        "--drop-unmeasured",
        action="store_true",
        help="also drop tasks with no rollouts in the store (default: keep them)",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    records = TraceStore(args.store).load()
    if not records:
        raise SystemExit(f"no rollouts in {args.store} -- run the baseline first")

    stats = pass_rates(records, config_id=args.config_id)
    result = filter_tasks(
        tasks, stats, lo=args.lo, hi=args.hi, keep_unmeasured=not args.drop_unmeasured
    )

    print(f"tasks={len(tasks)}  records={len(records)}  band=[{args.lo:.0%}, {args.hi:.0%}]")
    print(result.summary())

    if result.too_easy:
        print(f"  too easy (>{args.hi:.0%}): {', '.join(t.task.task_id for t in result.too_easy[:8])}")
    if result.too_hard:
        print(f"  too hard (<{args.lo:.0%}): {', '.join(t.task.task_id for t in result.too_hard[:8])}")
    if result.unmeasured:
        print(f"  unmeasured: {', '.join(t.task.task_id for t in result.unmeasured[:8])}")

    if not result.kept:
        raise SystemExit(
            "nothing left in the band: the task set is too easy or too hard for this model. "
            "Fix the difficulty, do not tune the learning rate."
        )

    expected = {t.task.task_id: t.expected for t in result.kept}
    splits = {t.task.task_id: t.split for t in result.kept}
    meta = {t.task.task_id: (t.meta or {}) for t in result.kept}
    count = dump_tasks(args.out, [t.task for t in result.kept], expected, splits=splits, meta=meta)

    print()
    print(f"wrote {count} tasks -> {args.out}")
    print(f"next: point the training config's tasks_file at {args.out}")
    (args.out.parent / (args.out.stem + "-report.json")).write_text(
        json.dumps(
            {
                "source_tasks": str(args.tasks),
                "store": str(args.store),
                "band": [args.lo, args.hi],
                "kept": [t.task.task_id for t in result.kept],
                "too_easy": [t.task.task_id for t in result.too_easy],
                "too_hard": [t.task.task_id for t in result.too_hard],
                "unmeasured": [t.task.task_id for t in result.unmeasured],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
