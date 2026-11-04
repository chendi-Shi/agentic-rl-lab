from __future__ import annotations

import pytest

from arlab.rollout import RunnerConfig, run_task
from arlab.store import TraceStore
from arlab.tasks.lookup_tasks import (
    LOOKUP_ORDERS,
    LookupEnv,
    make_lookup_tasks,
    scripted_policies,
)


def test_every_task_is_unique_and_has_a_plan():
    tasks = make_lookup_tasks()
    ids = [item.task.task_id for item in tasks]

    assert len(tasks) >= 60
    assert len(set(ids)) == len(ids)
    assert all(item.plan for item in tasks)


def test_answers_need_the_tool_not_the_prompt():
    """No amount appears in the instruction, so the answer cannot be guessed."""
    amounts = [str(value) for value in LOOKUP_ORDERS.values()]
    for item in make_lookup_tasks():
        leaked = [a for a in amounts if a in item.task.instruction]
        assert not leaked, f"{item.task.task_id} leaks the amount in its prompt"


def test_calc_tool_is_required_and_works():
    env = LookupEnv()
    assert "calc" in env.tool_names()
    assert env.call("calc", {"expression": "199.0*7.5/10"}) == "149.25"
    # the order table still works alongside it
    assert "A123" in env.call("orders.get", {"order_id": "A123"})


def test_calc_rejects_escapes():
    env = LookupEnv()
    with pytest.raises(ValueError):
        env.call("calc", {"expression": "__import__('os')"})


def test_scripted_plans_score_full_marks(tmp_path):
    """Oracle + environment agree, and the plans actually chain two tools."""

    store = TraceStore(tmp_path / "t.jsonl")
    plans = scripted_policies()
    cfg = RunnerConfig(config_id="lookup", seed=0, oracle="answer")

    for item in make_lookup_tasks():
        plan = plans[item.task.task_id]

        def policy(task, steps, plan=plan):
            index = len(steps)
            return plan[index] if index < len(plan) else None

        record = run_task(
            item.task, LookupEnv, policy, cfg, policy_name="scripted", store=store
        )
        assert record.reward["r_task"] == pytest.approx(1.0), item.task.task_id


def test_plans_use_both_tools():
    for item in make_lookup_tasks():
        names = {getattr(step, "name", None) for step in item.plan}
        assert "orders.get" in names
        assert "calc" in names, f"{item.task.task_id} can be solved without a calculator"
