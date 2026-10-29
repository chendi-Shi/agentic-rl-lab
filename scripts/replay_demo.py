"""End-to-end CPU demo: scripted policies -> verifiable reward -> diagnostics.

No GPU, no framework. This proves the reward actually discriminates before any
money is spent on renting a card.

    python scripts/replay_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.diagnostics import (  # noqa: E402
    classify_difficulty,
    detect_advantage_collapse,
    length_percentiles,
)
from arlab.reward import compute_reward  # noqa: E402
from arlab.tasks import Step, SyntheticBusinessEnv, Task, ToolCall, Trajectory  # noqa: E402

TASKS = [
    Task(
        task_id="refund-A123",
        instruction="订单 A123 未发货，全额退款。",
        required_fields={"A123.refunded": True, "A123.refund_amount": 199.0},
        max_steps=8,
        tools=SyntheticBusinessEnv.TOOLS,
    ),
    Task(
        task_id="refund-B456",
        instruction="订单 B456 已发货，若客户要求则退款。",
        required_fields={"B456.refunded": True, "B456.refund_amount": 89.5},
        max_steps=8,
        tools=SyntheticBusinessEnv.TOOLS,
    ),
    Task(
        task_id="reject-C789",
        instruction="订单 C789 已签收，不需要退款，确认状态即可。",
        required_fields={
            "C789.refunded": False,
            "C789.status": "delivered",
            "C789.last_queried": True,  # evidence of inspection; blocks "do nothing"
        },
        max_steps=8,
        tools=SyntheticBusinessEnv.TOOLS,
    ),
]

ORDERS = {"refund-A123": "A123", "refund-B456": "B456", "reject-C789": "C789"}


def policy_correct(order_id: str) -> list[tuple[str, dict]]:
    calls = [("orders.get", {"order_id": order_id})]
    if order_id != "C789":
        calls.append(("orders.refund", {"order_id": order_id}))
    return calls


def policy_claim_only(order_id: str) -> list[tuple[str, dict]]:
    return []  # model just says "done" without touching a tool


def policy_get_only(order_id: str) -> list[tuple[str, dict]]:
    return [("orders.get", {"order_id": order_id})]


def policy_spam(order_id: str) -> list[tuple[str, dict]]:
    return [("orders.get", {"order_id": order_id})] * 6


POLICIES = {
    "correct": policy_correct,
    "claim_only": policy_claim_only,
    "get_only": policy_get_only,
    "spam": policy_spam,
}


def run_one(task: Task, policy) -> tuple[object, int]:
    env = SyntheticBusinessEnv()
    order_id = ORDERS[task.task_id]
    traj = Trajectory(task_id=task.task_id)

    for name, args in policy(order_id):
        try:
            obs = env.call(name, args)
        except KeyError as exc:
            obs = f"ERROR: {exc}"
        traj.steps.append(Step(tool_call=ToolCall(name, args), observation=obs))

    if not traj.steps:
        traj.steps.append(Step(note="claiming success without calling tools"))
        traj.final_answer = "已完成"

    return compute_reward(task, traj, env.snapshot()), len(traj)


def main() -> int:
    print("=" * 78)
    print("reward breakdown by (task, policy)")
    print("=" * 78)
    print(f"{'task':<13}{'policy':<12}{'total':>8}{'r_task':>8}{'r_fmt':>8}{'matched':>9}{'steps':>7}")
    print("-" * 78)

    rewards_by_task: dict[str, list[float]] = {t.task_id: [] for t in TASKS}
    lengths: list[int] = []

    for task in TASKS:
        for name, policy in POLICIES.items():
            rb, n_steps = run_one(task, policy)
            rewards_by_task[task.task_id].append(rb.total)
            lengths.append(n_steps)
            print(
                f"{task.task_id:<13}{name:<12}{rb.total:>8.3f}{rb.r_task:>8.2f}"
                f"{rb.r_format:>8.2f}{rb.matched_fields}/{rb.total_fields:<7}{n_steps:>7}"
            )
        print("-" * 78)

    print()
    print("=" * 78)
    print("diagnostics")
    print("=" * 78)
    collapse = detect_advantage_collapse(rewards_by_task)
    print(collapse.summary())
    if collapse.collapsed_task_ids:
        print(f"  collapsed: {', '.join(collapse.collapsed_task_ids)}")

    diff = classify_difficulty(rewards_by_task)
    print(f"  keep    : {diff.kept or '-'}")
    print(f"  too easy: {diff.too_easy or '-'}")
    print(f"  too hard: {diff.too_hard or '-'}")

    tail = length_percentiles(lengths)
    print(
        f"  length  : p50={tail['p50']:.0f} p90={tail['p90']:.0f} p99={tail['p99']:.0f} "
        f"tail_share={tail['tail_share']:.1%}"
    )

    print()
    print("=" * 78)
    print("what this proves")
    print("=" * 78)
    print("- claim_only scores 0 on r_task: the oracle reads environment state only.")
    print("- get_only earns partial credit, so a group never collapses to all-zero by design.")
    print("- spam is penalised by duplicate + step penalties.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
