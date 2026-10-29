"""Scripted reference policies for the built-in task sets.

**These are not baselines.** They replay the correct actions and therefore score
near-perfect on both task families. Their only job is to prove that a task set,
an environment, the reward and the store work end to end before a real model is
involved. A number produced by a scripted policy must never be reported as a
model result.
"""

from __future__ import annotations

import json

from arlab.rollout.runner import FinalAnswer
from arlab.tasks.business_tasks import ORDER_OF, make_business_tasks
from arlab.tasks.math_env import make_math_tasks
from arlab.tasks.schema import Task, ToolCall

_MATH_CODE = {
    "m-01": "17*23+9",
    "m-02": "(144/12)**2-19",
    "m-03": "sum(range(1,51))",
    "m-04": "2**10-24",
    "m-05": "(12-2)*180",
    "m-06": "6*5*4*3*2*1+5*4*3*2*1",
}


def _build() -> dict[str, list]:
    plans: dict[str, list] = {}
    for task in make_math_tasks():
        plans[task.task_id] = [
            ToolCall("python", {"code": _MATH_CODE[task.task_id]}),
            FinalAnswer(task.required_fields["answer"]),
        ]
    for task in make_business_tasks():
        order = ORDER_OF[task.task_id]
        plan = [ToolCall("orders.get", {"order_id": order})]
        if not task.task_id.startswith("reject"):
            plan.append(ToolCall("orders.refund", {"order_id": order}))
        plans[task.task_id] = plan
    return plans


SCRIPTED_PLANS: dict[str, list] = _build()


def scripted_policy(task: Task, steps):
    """Replay the planned actions for a built-in task."""

    plan = SCRIPTED_PLANS.get(task.task_id)
    if plan is None:
        raise KeyError(
            f"no scripted plan for {task.task_id!r}; use --policy llm for real tasks"
        )
    index = len(steps)
    return plan[index] if index < len(plan) else None


def scripted_diagnosis_policy(task: Task, steps):
    """Stage-2 reference: use all three MCP tools, then submit the oracle answer.

    Only exists to prove the MCP path and the structured reward work. It submits
    ``required_fields`` verbatim, so a perfect score here is a tautology.
    """

    service = task.required_fields.get("service") or task.task_id
    plan = [
        ToolCall("service_metrics", {"service": service, "window_minutes": 30}),
        ToolCall("knowledge_search", {"query": "rollback error rate", "service": service}),
        ToolCall("incident_changes", {"service": service, "limit": 20}),
        FinalAnswer(json.dumps(dict(task.required_fields), ensure_ascii=False)),
    ]
    index = len(steps)
    return plan[index] if index < len(plan) else None


def scripted_policy_for(task: Task, steps):
    """Pick the reference policy that matches the task's family."""

    if task.family in {"math", "business", "default"}:
        return scripted_policy(task, steps)
    return scripted_diagnosis_policy(task, steps)
