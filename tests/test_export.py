from __future__ import annotations

import json

import pytest

from arlab.tasks.math_env import make_math_tasks
from arlab.tasks.stage_io import LoadedTask, dump_tasks, load_tasks
from arlab.train.export import REQUIRED_KEYS, export_jsonl, to_verl_record, validate_record


def loaded(task_id: str = "m-01", answer: str = "400") -> LoadedTask:
    tasks = {t.task_id: t for t in make_math_tasks()}
    task = tasks[task_id]
    return LoadedTask(task=task, expected={"answer": answer}, split="dev")


def test_record_has_every_required_key():
    record = to_verl_record(loaded())
    assert all(key in record for key in REQUIRED_KEYS)
    assert validate_record(record) == []


def test_prompt_shape_and_oracle_travels():
    record = to_verl_record(loaded())

    roles = [message["role"] for message in record["prompt"]]
    assert roles == ["system", "user"]
    assert "python" in record["prompt"][0]["content"]
    assert json.loads(record["reward_model"]["ground_truth"]) == {"answer": "400"}
    assert record["extra_info"]["task_id"] == "m-01"


def test_validation_catches_missing_keys():
    problems = validate_record({"prompt": []})
    assert any("missing keys" in p for p in problems)


def test_validation_catches_bad_prompt_messages():
    record = to_verl_record(loaded())
    record["prompt"] = [{"role": "system"}]
    problems = validate_record(record)
    assert any("needs 'role' and 'content'" in p for p in problems)


def test_validation_catches_missing_ground_truth():
    record = to_verl_record(loaded())
    record["reward_model"] = {"style": "rule"}
    assert any("ground_truth" in p for p in validate_record(record))


def test_export_writes_one_valid_line_per_task(tmp_path):
    tasks = [loaded("m-01", "400"), loaded("m-02", "125")]
    path = tmp_path / "train.jsonl"

    count = export_jsonl(tasks, path)

    assert count == 2
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert validate_record(json.loads(line)) == []


def test_export_roundtrips_through_stage_io(tmp_path):
    math_tasks = make_math_tasks()[:2]
    expected = {t.task_id: {"answer": t.required_fields["answer"]} for t in math_tasks}
    frozen = tmp_path / "stage.json"
    dump_tasks(frozen, math_tasks, expected)

    loaded_tasks = load_tasks(frozen)
    out = tmp_path / "verl.jsonl"
    export_jsonl(loaded_tasks, out)

    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["extra_info"]["task_id"] for r in records] == [t.task_id for t in math_tasks]
    assert all(validate_record(r) == [] for r in records)
