from __future__ import annotations

import pytest

from arlab.store import TraceRecord
from arlab.tasks.math_env import make_math_tasks
from arlab.tasks.schema import Trajectory
from arlab.tasks.stage_io import LoadedTask
from arlab.train.filtering import filter_tasks, pass_rates


def record(task_id: str, r_task: float, config_id: str = "base") -> TraceRecord:
    return TraceRecord(
        key=f"{task_id}|p|{config_id}|0",
        task_id=task_id,
        policy="p",
        config_id=config_id,
        seed=0,
        trajectory=Trajectory(task_id=task_id),
        reward={"total": r_task, "r_task": r_task},
    )


def loaded(tasks, count: int = 4) -> list[LoadedTask]:
    return [LoadedTask(task=t, expected=dict(t.required_fields)) for t in tasks[:count]]


def test_pass_rate_uses_r_task_not_total():
    """Partial credit gives every rollout a positive total; pass means solved."""

    stats = pass_rates([record("a", 1.0), record("a", 0.6), record("a", 0.0)])
    assert stats["a"].pass_rate == pytest.approx(1 / 3)


def test_config_filter_is_respected():
    records = [record("a", 1.0, "base"), record("a", 0.0, "rl")]
    assert pass_rates(records, config_id="base")["a"].pass_rate == pytest.approx(1.0)
    assert pass_rates(records, config_id="rl")["a"].pass_rate == pytest.approx(0.0)


def test_bands_split_correctly():
    tasks = loaded(make_math_tasks(), 4)
    stats = pass_rates(
        [
            record(tasks[0].task.task_id, 1.0),   # always solved -> too easy
            record(tasks[1].task.task_id, 0.0),   # never solved  -> too hard
            record(tasks[2].task.task_id, 1.0),   # 50%           -> keep
            record(tasks[2].task.task_id, 0.0),
        ]
    )

    result = filter_tasks(tasks, stats, lo=0.2, hi=0.8)

    # tasks[2] is measured and in band; tasks[3] has no rollouts -> unmeasured,
    # kept by default so a partial baseline never silently deletes tasks.
    assert [t.task.task_id for t in result.kept] == [
        tasks[2].task.task_id,
        tasks[3].task.task_id,
    ]
    assert [t.task.task_id for t in result.unmeasured] == [tasks[3].task.task_id]
    assert [t.task.task_id for t in result.too_easy] == [tasks[0].task.task_id]
    assert [t.task.task_id for t in result.too_hard] == [tasks[1].task.task_id]
    assert result.dropped == 2


def test_unmeasured_kept_by_default():
    tasks = loaded(make_math_tasks(), 2)
    result = filter_tasks(tasks, {}, lo=0.2, hi=0.8)

    assert len(result.kept) == 2
    assert len(result.unmeasured) == 2


def test_unmeasured_can_be_dropped():
    tasks = loaded(make_math_tasks(), 2)
    result = filter_tasks(tasks, {}, lo=0.2, hi=0.8, keep_unmeasured=False)

    assert result.kept == []
    assert len(result.unmeasured) == 2


def test_bad_band_rejected():
    with pytest.raises(ValueError):
        filter_tasks([], {}, lo=0.8, hi=0.2)


def test_boundaries_are_inclusive_inside_the_band():
    tasks = loaded(make_math_tasks(), 1)
    task_id = tasks[0].task.task_id
    # exactly 0.2 -> kept (not "too hard"), exactly 0.8 -> kept (not "too easy")
    stats = pass_rates([record(task_id, 1.0)] + [record(task_id, 0.0)] * 4)
    assert filter_tasks(tasks, stats, lo=0.2, hi=0.8).kept == tasks
