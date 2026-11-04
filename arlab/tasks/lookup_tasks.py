"""Tasks that require chaining two tools: look the number up, then compute.

Two earlier designs failed for instructive reasons:

* pure arithmetic (even three-hop word problems) -> 48/48 solved, because the
  whole thing collapses into one ``python`` call and the calculator does the work.
* lookup without a calculator -> 0% on the discount tasks, because a 4B model
  cannot do ``199.0 * 0.75`` in its head and simply echoed the amount back.

So these tasks need ``orders.get`` to fetch the value *and* ``calc`` to use it.
The answer stays machine-checkable, but it cannot be produced in one shot and it
cannot be produced by mental arithmetic either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from arlab.rollout.runner import FinalAnswer
from arlab.tasks.env import SyntheticBusinessEnv
from arlab.tasks.math_env import ArithmeticExecutor
from arlab.tasks.schema import Task, ToolCall

TOOLS = ("orders.get", "calc", "orders.refund", "shipping.track")

#: A wider order table than the 3-order demo env: task count scales as
#: ``N*5 + C(N,2)*3``, and 24 tasks only yielded 8 in the learnable band.
LOOKUP_ORDERS: dict[str, float] = {
    "A123": 199.0,
    "B456": 89.5,
    "C789": 320.0,
    "D012": 1204.0,
    "E345": 78.25,
    "F678": 2450.5,
}


class LookupEnv(SyntheticBusinessEnv):
    """The order table plus a safe calculator."""

    TOOLS = TOOLS
    ORDERS = {
        oid: {"status": "paid", "amount": amount, "refunded": False, "refund_amount": 0.0}
        for oid, amount in LOOKUP_ORDERS.items()
    }

    def __init__(self) -> None:
        super().__init__()
        self._calc = ArithmeticExecutor()

    def tool_names(self) -> tuple[str, ...]:
        return self.TOOLS

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        if name == "calc":
            expression = args.get("expression") or args.get("code")
            if not isinstance(expression, str) or not expression.strip():
                raise ValueError("calc requires a non-empty 'expression' string")
            return self._calc.run(expression)
        return super().call(name, args)


@dataclass(frozen=True)
class LookupTask:
    task: Task
    plan: list  # scripted actions, for offline verification of the oracle


def _amount(order_id: str) -> float:
    return float(LOOKUP_ORDERS[order_id])


def _fmt(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _build(
    task_id: str,
    instruction: str,
    answer: str,
    order_ids: tuple[str, ...],
    expression: str,
) -> LookupTask:
    task = Task(
        task_id=task_id,
        instruction=instruction
        + " 先用 orders.get 查金额，再用 calc 计算，最后提交答案（只填数字）。",
        required_fields={"answer": answer},
        max_steps=8,
        tools=TOOLS,
        family="lookup",
    )
    plan: list = [ToolCall("orders.get", {"order_id": oid}) for oid in order_ids]
    plan.append(ToolCall("calc", {"expression": expression}))
    plan.append(FinalAnswer(answer))
    return LookupTask(task=task, plan=plan)


def make_lookup_tasks() -> list[LookupTask]:
    """Five operations per order plus three per order pair."""

    out: list[LookupTask] = []
    ids = tuple(LOOKUP_ORDERS)
    discounts = (7.5, 8.0, 8.5, 9.0)

    for oid in ids:
        amount = _amount(oid)
        for off in discounts:
            out.append(
                _build(
                    f"lk-disc-{oid}-{str(off).replace('.', '')}",
                    f"订单 {oid} 的金额按 {off}/10 付款，应付多少元？",
                    _fmt(round(amount * off / 10, 2)),
                    (oid,),
                    f"{amount}*{off}/10",
                )
            )
        out.append(
            _build(
                f"lk-third-{oid}",
                f"订单 {oid} 金额的三分之一是多少？（保留两位小数）",
                _fmt(round(amount / 3, 2)),
                (oid,),
                f"round({amount}/3, 2)",
            )
        )

    for i, first in enumerate(ids):
        for second in ids[i + 1 :]:
            a, b = _amount(first), _amount(second)
            out.append(
                _build(
                    f"lk-sum-{first}-{second}",
                    f"订单 {first} 和订单 {second} 的金额合计多少元？",
                    _fmt(round(a + b, 2)),
                    (first, second),
                    f"{a}+{b}",
                )
            )
            out.append(
                _build(
                    f"lk-diff-{first}-{second}",
                    f"订单 {first} 比订单 {second} 贵多少元？（用较大减较小）",
                    _fmt(round(abs(a - b), 2)),
                    (first, second),
                    f"abs({a}-{b})",
                )
            )
            out.append(
                _build(
                    f"lk-sumdisc-{first}-{second}",
                    f"订单 {first} 和订单 {second} 的金额合计，再按 9/10 付款，应付多少元？",
                    _fmt(round((a + b) * 0.9, 2)),
                    (first, second),
                    f"round(({a}+{b})*0.9, 2)",
                )
            )
    return out


def as_dump() -> tuple[list[Task], dict[str, dict], dict[str, dict]]:
    """Shape the tasks for ``stage_io.dump_tasks``, with plans kept in meta."""

    built = make_lookup_tasks()
    tasks = [item.task for item in built]
    expected = {
        item.task.task_id: {"answer": item.task.required_fields["answer"]} for item in built
    }
    meta = {
        item.task.task_id: {
            "plan": [
                {"tool": step.name, "args": dict(step.args)}
                for step in item.plan
                if isinstance(step, ToolCall)
            ],
            "answer": item.task.required_fields["answer"],
        }
        for item in built
    }
    return tasks, expected, meta


def scripted_policies() -> dict[str, list]:
    """task_id -> scripted action plan, for offline verification of the oracle."""

    return {item.task.task_id: item.plan for item in make_lookup_tasks()}
