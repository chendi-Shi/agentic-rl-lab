from __future__ import annotations

from arlab.reward.answer import (
    answers_match,
    compute_answer_reward,
    compute_structured_answer_reward,
    extract_number,
)
from arlab.tasks.math_env import make_math_tasks
from arlab.tasks.schema import Step, Task, ToolCall, Trajectory


def answer_task(answer: str, max_steps: int = 6) -> Task:
    return Task(
        task_id="m",
        instruction="compute",
        required_fields={"answer": answer},
        max_steps=max_steps,
        tools=("python",),
    )


def test_numeric_equivalence_and_prose_extraction():
    assert answers_match("400", "400.0")
    assert answers_match("结果是 400", "400")
    assert not answers_match("401", "400")
    assert extract_number("the result is 1,234") == "234"  # comma split, documented


def test_correct_answer_scores_full(tmp_path):
    task = answer_task("400")
    traj = Trajectory(
        task_id="m",
        steps=[Step(tool_call=ToolCall("python", {"code": "17*23+9"}), observation="400")],
        final_answer="400",
    )

    rb = compute_answer_reward(task, traj)

    assert rb.r_task == 1.0
    assert rb.total > 1.0


def test_wrong_answer_scores_zero_task_reward():
    task = answer_task("400")
    traj = Trajectory(task_id="m", final_answer="42")

    rb = compute_answer_reward(task, traj)

    assert rb.r_task == 0.0


def test_missing_answer_scores_zero():
    task = answer_task("400")
    rb = compute_answer_reward(task, Trajectory(task_id="m"))
    assert rb.r_task == 0.0


def test_truncated_episode_voids_the_answer():
    task = answer_task("400", max_steps=1)
    traj = Trajectory(
        task_id="m",
        steps=[Step(tool_call=ToolCall("python", {"code": "1"}), observation="1")],
        final_answer="400",
    )
    # len(steps) == max_steps, so not truncated yet
    assert compute_answer_reward(task, traj).r_task == 1.0

    traj.steps.append(Step(tool_call=ToolCall("python", {"code": "2"}), observation="2"))
    assert compute_answer_reward(task, traj).r_task == 0.0


def test_structured_answer_gets_dense_partial_credit():
    task = Task(
        task_id="diag",
        instruction="diagnose",
        required_fields={"service": "payments"},
        max_steps=10,
        tools=("knowledge_search", "service_metrics", "incident_changes"),
    )
    expected = {
        "service": "payments",
        "status": "incident",
        "error_rate": 0.42,
        "citations": ["ev-1", "ev-2"],
    }
    traj = Trajectory(
        task_id="diag",
        final_answer='{"service": "payments", "status": "healthy", "error_rate": 0.43, "citations": ["ev-1"]}',
    )

    rb = compute_structured_answer_reward(task, traj, expected)

    # service exact, error_rate within 1%, citations 1 of 2, status wrong.
    assert rb.r_task > 0.3
    assert rb.r_task < 1.0


def test_unparseable_structured_answer_scores_zero():
    task = Task(task_id="diag", instruction="x", required_fields={"a": 1}, tools=("t",))
    traj = Trajectory(task_id="diag", final_answer="not json")

    rb = compute_structured_answer_reward(task, traj, {"a": 1})

    assert rb.r_task == 0.0


def test_math_task_family_is_well_formed():
    tasks = make_math_tasks()
    assert len(tasks) >= 4
    assert all(t.tools == ("python",) for t in tasks)
    assert len({t.task_id for t in tasks}) == len(tasks)
