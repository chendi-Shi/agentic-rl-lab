"""Stage-2 task family: refund / no-op decisions over the synthetic business env.

Kept separate from ``env.py`` so the environment (state machine) and the task set
(oracle targets) can evolve independently -- the real stage-2 set will be
generated from ``byte-agent-platform``'s 32 services instead.
"""

from __future__ import annotations

from arlab.tasks.env import SyntheticBusinessEnv
from arlab.tasks.schema import Task

#: task_id -> order id
ORDER_OF: dict[str, str] = {
    "refund-A123": "A123",
    "refund-B456": "B456",
    "reject-C789": "C789",
}


def make_business_tasks() -> list[Task]:
    return [
        Task(
            task_id="refund-A123",
            instruction="订单 A123 未发货，全额退款。",
            required_fields={"A123.refunded": True, "A123.refund_amount": 199.0},
            max_steps=8,
            tools=SyntheticBusinessEnv.TOOLS,
            family="business",
        ),
        Task(
            task_id="refund-B456",
            instruction="订单 B456 已发货，若客户要求则退款。",
            required_fields={"B456.refunded": True, "B456.refund_amount": 89.5},
            max_steps=8,
            tools=SyntheticBusinessEnv.TOOLS,
            family="business",
        ),
        Task(
            task_id="reject-C789",
            instruction="订单 C789 已签收，不需要退款，确认状态即可。",
            required_fields={
                "C789.refunded": False,
                "C789.status": "delivered",
                # Evidence of inspection: without this a no-op policy scores full marks.
                "C789.last_queried": True,
            },
            max_steps=8,
            tools=SyntheticBusinessEnv.TOOLS,
            family="business",
        ),
    ]
