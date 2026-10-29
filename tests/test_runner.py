from __future__ import annotations

import json

import pytest

from arlab.rollout import AlreadyDone, FinalAnswer, RunnerConfig, run_batch, run_task
from arlab.store import TraceStore, rollout_key
from arlab.tasks.business_tasks import ORDER_OF, make_business_tasks
from arlab.tasks.env import SyntheticBusinessEnv
from arlab.tasks.schema import Task, ToolCall


def good_policy(task: Task, steps):
    order = ORDER_OF[task.task_id]
    if not steps:
        return ToolCall("orders.get", {"order_id": order})
    if task.task_id.startswith("reject"):
        return None
    if len(steps) == 1:
        return ToolCall("orders.refund", {"order_id": order})
    return None


def loop_forever(task: Task, steps):
    return ToolCall("orders.get", {"order_id": ORDER_OF[task.task_id]})


def test_successful_rollout_is_persisted(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl")
    task = make_business_tasks()[0]
    cfg = RunnerConfig(config_id="c1", seed=0)

    record = run_task(task, SyntheticBusinessEnv, good_policy, cfg, policy_name="ok", store=store)

    assert record.reward["r_task"] == pytest.approx(1.0)
    assert record.reward["truncated"] is False
    assert store.is_done(record.key)
    assert len(store.load()) == 1


def test_rerunning_a_done_key_is_rejected(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl")
    task = make_business_tasks()[0]
    cfg = RunnerConfig(config_id="c1", seed=0)

    run_task(task, SyntheticBusinessEnv, good_policy, cfg, policy_name="ok", store=store)
    with pytest.raises(AlreadyDone):
        run_task(task, SyntheticBusinessEnv, good_policy, cfg, policy_name="ok", store=store)


def test_batch_skips_completed_work(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl")
    tasks = make_business_tasks()
    cfg = RunnerConfig(config_id="c1", seed=0)

    first = run_batch(tasks, SyntheticBusinessEnv, good_policy, cfg, policy_name="ok", store=store)
    second = run_batch(tasks, SyntheticBusinessEnv, good_policy, cfg, policy_name="ok", store=store)

    assert first.n_run == len(tasks)
    assert first.n_skipped == 0
    assert second.n_run == 0
    assert second.n_skipped == len(tasks)
    assert len(store.load()) == len(tasks)


def test_step_overrun_is_recorded_as_truncated(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl")
    task = make_business_tasks()[0]
    cfg = RunnerConfig(config_id="c1", seed=0)

    record = run_task(
        task, SyntheticBusinessEnv, loop_forever, cfg, policy_name="loop", store=store
    )

    assert record.reward["truncated"] is True
    assert record.reward["r_task"] == pytest.approx(0.0)
    assert len(record.trajectory) == task.max_steps + 1


def test_final_answer_ends_the_episode(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl")
    task = make_business_tasks()[0]

    record = run_task(
        task,
        SyntheticBusinessEnv,
        lambda task, steps: FinalAnswer("done"),
        RunnerConfig(config_id="c1", seed=0, oracle="state"),
        policy_name="answer",
        store=store,
    )

    assert record.trajectory.final_answer == "done"
    assert len(record.trajectory) == 0


def test_failing_tool_is_captured_as_observation(tmp_path):
    store = TraceStore(tmp_path / "t.jsonl")
    task = make_business_tasks()[0]

    record = run_task(
        task,
        SyntheticBusinessEnv,
        lambda task, steps: ToolCall("orders.get", {"order_id": "ZZZ"}),
        RunnerConfig(config_id="c1", seed=0),
        policy_name="bad_tool",
        store=store,
    )

    assert record.trajectory.steps[0].observation.startswith("ERROR:")


def test_unknown_oracle_rejected():
    with pytest.raises(ValueError):
        RunnerConfig(oracle="nope")


def test_key_includes_every_dimension():
    assert rollout_key("t", "p", "c", 3) == "t|p|c|3"
    assert rollout_key("t", "p", "c", 3) != rollout_key("t", "p", "c", 4)
