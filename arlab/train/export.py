"""Export frozen tasks into the record shape verl's trainer reads.

verl's PPO/GRPO entry point consumes a dataset of records with a stable set of
keys. Getting this wrong shows up as a cryptic failure inside the trainer long
after the card is billed, so the schema is validated here, offline.

The record shape (matching the RLHFTrainer-style datasets verl builds on)::

    prompt          list[{"role", "content"}]
    data_source     str, used as the tag for the reward function
    ability         str
    reward_model    {"style": "rule", "ground_truth": ...}
    extra_info      dict, free-form; task id and oracle target live here
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from arlab.policy.openai import system_prompt
from arlab.tasks.stage_io import LoadedTask

REQUIRED_KEYS = ("prompt", "data_source", "ability", "reward_model", "extra_info")


def to_verl_record(loaded: LoadedTask, data_source: str = "arlab") -> dict[str, Any]:
    task = loaded.task
    system = system_prompt(task.tools)
    return {
        "prompt": [
            {"role": "system", "content": system},
            {"role": "user", "content": task.instruction},
        ],
        "data_source": data_source,
        "ability": task.family,
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(loaded.expected, ensure_ascii=False),
        },
        "extra_info": {
            "task_id": task.task_id,
            "split": loaded.split,
            "tools": list(task.tools),
            "max_steps": task.max_steps,
            "expected": loaded.expected,
        },
    }


def validate_record(record: Mapping[str, Any]) -> list[str]:
    """Return a list of problems; empty means the record is well-formed."""

    problems: list[str] = []
    missing = [key for key in REQUIRED_KEYS if key not in record]
    if missing:
        problems.append(f"missing keys: {', '.join(missing)}")

    prompt = record.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        problems.append("prompt must be a non-empty list")
    else:
        for index, message in enumerate(prompt):
            if not isinstance(message, dict) or {"role", "content"} - set(message):
                problems.append(f"prompt[{index}] needs 'role' and 'content'")

    reward_model = record.get("reward_model")
    if not isinstance(reward_model, dict) or "ground_truth" not in reward_model:
        problems.append("reward_model.ground_truth missing")

    extra = record.get("extra_info")
    if not isinstance(extra, dict) or "task_id" not in extra:
        problems.append("extra_info.task_id missing")
    return problems


def export_jsonl(tasks: Iterable[LoadedTask], path: str | Path, data_source: str = "arlab") -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as handle:
        for loaded in tasks:
            record = to_verl_record(loaded, data_source)
            problems = validate_record(record)
            if problems:
                raise ValueError(f"{loaded.task.task_id}: {'; '.join(problems)}")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count
