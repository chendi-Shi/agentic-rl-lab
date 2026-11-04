"""Export a frozen task set into verl's dataset format.

    python scripts/export_verl.py --tasks tasks/stage1.json --out data/stage1.jsonl

Every record is validated before writing: a schema mistake here fails deep inside
the trainer, after the card is already billing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.tasks.stage_io import load_tasks  # noqa: E402
from arlab.train.export import export_jsonl, validate_record  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-source", default="arlab")
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--plain-system",
        action="store_true",
        help="short system prompt; use with verl's tool_agent, which injects the real tool schemas",
    )
    args = parser.parse_args()

    loaded = load_tasks(args.tasks, split=args.split)
    if not loaded:
        raise SystemExit("no tasks matched")

    if args.plain_system:
        from arlab.train.export import PLAIN_SYSTEM

        for item in loaded:
            item.expected = dict(item.expected)
        records = [to_verl_record(item, args.data_source, system=PLAIN_SYSTEM) for item in loaded]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for record in records:
                problems = validate_record(record)
                if problems:
                    raise SystemExit(f"{record['extra_info']['task_id']}: {problems}")
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        count = len(records)
    else:
        count = export_jsonl(loaded, args.out, data_source=args.data_source)

    first = json.loads(args.out.read_text(encoding="utf-8").splitlines()[0])
    problems = validate_record(first)
    if problems:
        raise SystemExit(f"first record invalid: {problems}")

    print(f"wrote {count} records -> {args.out}  ({args.out.stat().st_size} bytes)")
    print(f"  data_source : {first['data_source']}")
    print(f"  ability     : {first['ability']}")
    print(f"  task_id     : {first['extra_info']['task_id']}")
    print(f"  ground truth: {first['reward_model']['ground_truth'][:80]}")
    print()
    print("Point verl at this file with data.train_files=<path>.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
