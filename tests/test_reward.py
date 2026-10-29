from __future__ import annotations

import pytest

from arlab.reward import compute_reward
from arlab.tasks import Step, SyntheticBusinessEnv, Task, ToolCall, Trajectory


def make_task(fields: dict, max_steps: int = 12) -> Task:
    return Task(
        task_id="t1",
        instruction="refund if needed",
        required_fields=fields,
        max_steps=max_steps,
        tools=SyntheticBusinessEnv.TOOLS,
    )


def run(env: SyntheticBusinessEnv, calls: list[tuple[str, dict]]) -> tuple[Trajectory, dict]:
    traj = Trajectory(task_id="t1")
    for name, args in calls:
        obs = env.call(name, args)
        traj.steps.append(Step(tool_call=ToolCall(name, args), observation=obs))
    return traj, env.snapshot()


def test_correct_refund_gets_full_task_reward():
    env = SyntheticBusinessEnv()
    task = make_task({"A123.refunded": True, "A123.refund_amount": 199.0})
    traj, state = run(env, [("orders.get", {"order_id": "A123"}), ("orders.refund", {"order_id": "A123"})])

    rb = compute_reward(task, traj, state)

    assert rb.r_task == pytest.approx(1.0)
    assert rb.matched_fields == 2
    assert rb.r_format == pytest.approx(1.0)
    assert rb.truncated is False
    # 1.0 task + 0.1 * 1.0 format - 0.02 step penalty
    assert rb.total == pytest.approx(1.08)


def test_claiming_without_calling_scores_zero_task_reward():
    """The oracle reads environment state, so prose cannot earn reward."""
    env = SyntheticBusinessEnv()
    task = make_task({"A123.refunded": True})
    traj = Trajectory(task_id="t1", steps=[Step(note="I refunded A123")], final_answer="已退款")

    rb = compute_reward(task, traj, env.snapshot())

    assert rb.r_task == pytest.approx(0.0)
    assert rb.n_calls == 0
    assert rb.r_format == pytest.approx(0.0)


def test_partial_credit_is_dense():
    """One of two required fields satisfied -> 0.5, not 0. Dense signal keeps GRPO alive."""
    env = SyntheticBusinessEnv()
    task = make_task({"A123.status": "paid", "A123.refunded": True})
    traj, state = run(env, [("orders.get", {"order_id": "A123"})])

    rb = compute_reward(task, traj, state)

    assert rb.matched_fields == 1
    assert rb.total_fields == 2
    assert rb.r_task == pytest.approx(0.5)


def test_invalid_tool_is_penalised_and_earns_no_format_credit():
    env = SyntheticBusinessEnv()
    task = make_task({"A123.refunded": True})
    traj = Trajectory(
        task_id="t1",
        steps=[Step(tool_call=ToolCall("orders.delete", {"order_id": "A123"}), observation="ERROR: nope")],
    )

    rb = compute_reward(task, traj, env.snapshot())

    assert rb.p_invalid == pytest.approx(0.2)
    assert rb.r_format == pytest.approx(0.0)


def test_repeated_identical_call_is_penalised():
    env = SyntheticBusinessEnv()
    task = make_task({"A123.refunded": True})
    traj, state = run(
        env,
        [("orders.get", {"order_id": "A123"}), ("orders.get", {"order_id": "A123"})],
    )

    rb = compute_reward(task, traj, state)

    assert rb.p_duplicate == pytest.approx(0.1)


def test_exceeding_step_budget_counts_as_failure():
    env = SyntheticBusinessEnv()
    task = make_task({"A123.refunded": True}, max_steps=2)
    traj, state = run(
        env,
        [
            ("orders.get", {"order_id": "A123"}),
            ("orders.get", {"order_id": "A123"}),
            ("orders.refund", {"order_id": "A123"}),
        ],
    )

    rb = compute_reward(task, traj, state)

    assert rb.truncated is True
    assert rb.r_task == pytest.approx(0.0)
    assert rb.p_truncation == pytest.approx(0.5)


def test_no_op_task_cannot_be_satisfied_by_doing_nothing():
    """Regression: a 'confirm only' task must require evidence of inspection.

    Without ``last_queried`` the task is satisfied by the initial environment
    state, so a policy that never calls a tool scores full marks.
    """
    env = SyntheticBusinessEnv()
    task = make_task(
        {"C789.status": "delivered", "C789.last_queried": True},
    )
    traj = Trajectory(task_id="t1", final_answer="已确认")

    rb = compute_reward(task, traj, env.snapshot())

    assert rb.matched_fields == 1  # status only
    assert rb.r_task == pytest.approx(0.5)


def test_format_reward_decays_after_warmup():
    env = SyntheticBusinessEnv()
    task = make_task({"A123.refunded": True}, max_steps=20)
    traj = Trajectory(task_id="t1")
    for _ in range(6):
        traj.steps.append(
            Step(tool_call=ToolCall("orders.get", {"order_id": "A123"}), observation="ok")
        )

    rb = compute_reward(task, traj, env.snapshot())

    # All calls valid, but late steps are weighted 0.3 -> ratio stays 1.0.
    assert rb.r_format == pytest.approx(1.0)
