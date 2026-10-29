from __future__ import annotations

import pytest

from arlab.rollout import RunnerConfig
from arlab.rollout.partial import PartialRolloutDriver
from arlab.store import TraceStore
from arlab.tasks.business_tasks import make_business_tasks
from arlab.tasks.env import SyntheticBusinessEnv
from arlab.tasks.schema import Task, ToolCall


def length_policy(target: int):
    """Calls a tool until the episode has ``target`` steps, then finishes."""

    def policy(task: Task, steps):
        if len(steps) >= target:
            return None
        return ToolCall("orders.get", {"order_id": "A123"})

    return policy


def long_task(max_steps: int = 30) -> Task:
    return Task(
        task_id="long",
        instruction="take many steps",
        required_fields={"A123.status": "paid"},
        max_steps=max_steps,
        tools=SyntheticBusinessEnv.TOOLS,
    )


def drive(tmp_path, *, partial: bool, length: int = 9, budget: int = 3, max_rounds: int = 64):
    store = TraceStore(tmp_path / "t.jsonl")
    driver = PartialRolloutDriver(
        [long_task()],
        SyntheticBusinessEnv,
        length_policy(length),
        RunnerConfig(config_id="c", seed=0),
        policy_name="p",
        store=store,
        round_budget=budget,
        partial_rollout=partial,
        max_rounds=max_rounds,
    )
    return driver.run(), store


def test_partial_rollout_executes_each_step_once(tmp_path):
    stats, store = drive(tmp_path, partial=True)

    assert stats.finished == 1
    assert stats.dropped == 0
    assert stats.steps_executed == 9  # every step exactly once, no recompute
    assert stats.steps_wasted == 0
    assert stats.resumes == 2  # two carry-overs that did more work
    # 3 rounds of work + 1 round whose only action is the policy saying "done".
    assert stats.rounds == 4
    assert len(store.load()) == 1


def test_truncation_wastes_the_partial_episode(tmp_path):
    stats, store = drive(tmp_path, partial=False)

    assert stats.dropped == 1
    assert stats.finished == 0
    assert stats.steps_executed == 3
    assert stats.steps_wasted == 3
    assert stats.waste_fraction == pytest.approx(1.0)
    record = store.load()[0]
    assert record.meta["dropped"] is True


def test_short_episode_behaves_identically_in_both_modes(tmp_path):
    partial, _ = drive(tmp_path, partial=True, length=2)
    truncated, _ = drive(tmp_path, partial=False, length=2)

    assert partial.steps_executed == truncated.steps_executed == 2
    assert partial.finished == truncated.finished == 1
    assert truncated.dropped == 0


def test_round_cap_does_not_lose_episodes(tmp_path):
    """Force-finishing at max_rounds keeps the episode, it does not vanish."""
    stats, store = drive(tmp_path, partial=True, length=9, budget=3, max_rounds=1)

    assert stats.finished == 1
    assert stats.steps_executed == 3
    assert len(store.load()) == 1


def test_bad_round_budget_rejected(tmp_path):
    with pytest.raises(ValueError):
        PartialRolloutDriver([], SyntheticBusinessEnv, length_policy(2), round_budget=0)


def test_reward_is_computed_for_resumed_episodes(tmp_path):
    """A resumed episode must be scored on its real final state, not on a partial one."""
    store = TraceStore(tmp_path / "t.jsonl")

    def refund_policy(task: Task, steps):
        if not steps:
            return ToolCall("orders.get", {"order_id": "A123"})
        if len(steps) == 1:
            return ToolCall("orders.refund", {"order_id": "A123"})
        return None

    task = make_business_tasks()[0]
    driver = PartialRolloutDriver(
        [task],
        SyntheticBusinessEnv,
        refund_policy,
        RunnerConfig(config_id="c", seed=0),
        policy_name="p",
        store=store,
        round_budget=1,  # one step per round -> every step crosses a boundary
        partial_rollout=True,
    )
    stats = driver.run()
    record = store.load()[0]

    assert stats.resumes == 1
    assert record.reward["r_task"] == pytest.approx(1.0)
