from __future__ import annotations

import json

import pytest

from arlab.tasks.math_env import make_math_tasks
from arlab.tasks.stage_io import dump_tasks, load_tasks, task_ids


def test_dump_and_load_roundtrip(tmp_path):
    tasks = make_math_tasks()[:3]
    expected = {t.task_id: {"answer": t.required_fields["answer"]} for t in tasks}
    path = tmp_path / "stage.json"

    n = dump_tasks(path, tasks, expected)
    loaded = load_tasks(path)

    assert n == 3
    assert task_ids(loaded) == [t.task_id for t in tasks]
    assert loaded[0].expected == {"answer": tasks[0].required_fields["answer"]}
    assert loaded[0].task.tools == ("python",)


def test_split_filter(tmp_path):
    tasks = make_math_tasks()[:2]
    expected = {t.task_id: {"answer": t.required_fields["answer"]} for t in tasks}
    path = tmp_path / "stage.json"
    dump_tasks(path, tasks, expected, splits={tasks[0].task_id: "dev", tasks[1].task_id: "holdout"})

    assert [t.split for t in load_tasks(path, split="holdout")] == ["holdout"]
    assert len(load_tasks(path)) == 2


def test_missing_oracle_target_is_an_error(tmp_path):
    tasks = make_math_tasks()[:1]
    with pytest.raises(KeyError):
        dump_tasks(tmp_path / "x.json", tasks, {})


def test_expected_fields_become_required_fields(tmp_path):
    """The oracle target must travel with the task so the runner can score it."""
    tasks = make_math_tasks()[:1]
    path = tmp_path / "stage.json"
    dump_tasks(path, tasks, {tasks[0].task_id: {"answer": "400"}})

    loaded = load_tasks(path)[0]
    assert loaded.task.required_fields == {"answer": "400"}


def test_not_a_list_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_tasks(path)
