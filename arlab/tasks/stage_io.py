"""Task-set serialisation: frozen JSON in, typed tasks out.

The task set is frozen to disk so an experiment is reproducible after the
upstream corpus changes. Each record carries the oracle target with it, which is
what makes stage-2 (structured answers) runnable through the runner: the answer
oracle needs the expected fields, not just the environment state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from arlab.tasks.schema import Task


@dataclass
class LoadedTask:
    task: Task
    expected: dict[str, Any]
    split: str = "dev"
    meta: dict[str, Any] | None = None


def dump_tasks(
    path: str | Path,
    tasks: Iterable[Task],
    expected: Mapping[str, Mapping[str, Any]],
    splits: Mapping[str, str] | None = None,
    meta: Mapping[str, Mapping[str, Any]] | None = None,
) -> int:
    """Write a frozen task set. Returns the number of records written."""

    records: list[dict[str, Any]] = []
    for task in tasks:
        if task.task_id not in expected:
            raise KeyError(f"no oracle target for task {task.task_id}")
        records.append(
            {
                "task_id": task.task_id,
                "split": (splits or {}).get(task.task_id, "dev"),
                "family": task.family,
                "instruction": task.instruction,
                "expected": dict(expected[task.task_id]),
                "tools": list(task.tools),
                "max_steps": task.max_steps,
                "meta": dict((meta or {}).get(task.task_id, {})),
            }
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(records)


def load_tasks(path: str | Path, split: str | None = None) -> list[LoadedTask]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("task file must contain a list")

    out: list[LoadedTask] = []
    for record in payload:
        if split is not None and record.get("split") != split:
            continue
        expected = dict(record["expected"])
        task = Task(
            task_id=record["task_id"],
            instruction=record["instruction"],
            required_fields=expected,  # oracle target travels with the task
            max_steps=int(record.get("max_steps", 10)),
            tools=tuple(record.get("tools", ())),
            family=record.get("family", "default"),
        )
        out.append(
            LoadedTask(
                task=task,
                expected=expected,
                split=record.get("split", "dev"),
                meta=record.get("meta"),
            )
        )
    return out


def task_ids(tasks: Sequence[LoadedTask]) -> list[str]:
    return [t.task.task_id for t in tasks]
