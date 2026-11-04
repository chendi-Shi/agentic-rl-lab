"""Freeze a task set to JSON.

    python scripts/make_tasks.py --stage 1
    python scripts/make_tasks.py --stage 2 --services ../byte-agent-platform/examples/corpus/services.json

Stage 1 is the math/code-interpreter family. Stage 2 is generated from
``byte-agent-platform``'s service corpus, which already carries per-task labels,
so no human annotation is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.tasks.math_env import MathEnv, make_math_tasks  # noqa: E402
from arlab.tasks.stage_io import dump_tasks  # noqa: E402

#: Fields the stage-2 oracle grades. ``citations`` is excluded on purpose: it can
#: only be checked against live evidence ids from the MCP server, not the frozen
#: corpus, so grading it offline would produce a fake signal.
GRADED_FIELDS = ("service", "error_rate", "status", "likely_cause", "recommendation", "uncertainty")


def stage1(out: Path) -> int:
    tasks = make_math_tasks()
    expected = {t.task_id: {"answer": t.required_fields["answer"]} for t in tasks}
    return dump_tasks(out, tasks, expected)


def stage2(out: Path, services_json: Path) -> int:
    corpus = json.loads(services_json.read_text(encoding="utf-8"))
    raw = corpus.get("tasks")
    if not raw:
        raise SystemExit(f"no tasks in {services_json}")

    tools = ("knowledge_search", "service_metrics", "incident_changes")
    records = []
    for item in raw:
        expected = {}
        for field in GRADED_FIELDS:
            if field == "uncertainty":
                value = item.get("expected_uncertainty")
            else:
                value = item.get(field)
            if value is not None:
                expected[field] = value
        if len(expected) < 4:
            continue  # not enough labels to score meaningfully
        records.append(
            {
                "task_id": item["id"],
                "split": item.get("split", "dev"),
                "family": item.get("family", "diagnosis"),
                "instruction": item["prompt"],
                "expected": expected,
                "tools": list(item.get("required_tools") or tools),
                "max_steps": 12,
                "meta": {
                    "service": item.get("service"),
                    "source": item.get("source"),
                    "causal_source": item.get("causal_source"),
                    "query": item.get("query"),
                    "uncertainty_required": item.get("uncertainty_required"),
                    "thresholds": item.get("thresholds"),
                },
            }
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def stage1_generated(out: Path, per_tier: int, seed: int = 0) -> int:
    """A training-sized arithmetic set: ``per_tier`` tasks in each difficulty band."""

    from arlab.tasks.math_gen import generate, tier_counts, verify
    from arlab.tasks.math_env import MathEnv
    from arlab.tasks.schema import Task

    gen = generate(per_tier=per_tier, seed=seed)
    problems = verify(gen)
    if problems:
        raise SystemExit("generator produced invalid tasks:\n  " + "\n  ".join(problems[:5]))

    tasks = [
        Task(
            task_id=item.task_id,
            instruction=item.instruction,
            required_fields={"answer": item.answer},
            max_steps=6,
            tools=MathEnv.TOOLS,
            family="math",
        )
        for item in gen
    ]
    expected = {item.task_id: {"answer": item.answer} for item in gen}
    meta = {item.task_id: {"tier": item.tier, "code": item.code} for item in gen}
    count = dump_tasks(out, tasks, expected, meta=meta)
    print("tiers:", tier_counts(gen))
    return count


def stage3_lookup(out: Path) -> int:
    """Tasks whose numbers live behind a tool call, not in the prompt."""

    from arlab.tasks.lookup_tasks import as_dump

    tasks, expected, meta = as_dump()
    return dump_tasks(out, tasks, expected, meta=meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--generated", type=int, default=0, help="tasks per difficulty tier (stage 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--services",
        type=Path,
        default=Path("../byte-agent-platform/examples/corpus/services.json"),
    )
    args = parser.parse_args()

    if args.stage == 1:
        if args.generated:
            out = args.out or Path("tasks/stage1-train.json")
            n = stage1_generated(out, args.generated, seed=args.seed)
        else:
            out = args.out or Path("tasks/stage1.json")
            n = stage1(out)
    elif args.stage == 3:
        out = args.out or Path("tasks/stage3-lookup.json")
        n = stage3_lookup(out)
    else:
        out = args.out or Path("tasks/stage2.json")
        n = stage2(out, args.services)

    if args.stage == 2:
        corpus = json.loads(args.services.read_text(encoding="utf-8"))
        splits: dict[str, int] = {}
        for item in corpus["tasks"]:
            splits[item.get("split", "dev")] = splits.get(item.get("split", "dev"), 0) + 1
        print("corpus splits:", ", ".join(f"{k}={v}" for k, v in sorted(splits.items())))

    print(f"wrote {n} tasks -> {out}  ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
