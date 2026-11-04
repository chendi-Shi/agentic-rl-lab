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

    _OPS = {
        "multiply": "*",
        "mul": "*",
        "times": "*",
        "add": "+",
        "plus": "+",
        "subtract": "-",
        "minus": "-",
        "divide": "/",
        "div": "/",
        "power": "**",
        "pow": "**",
    }

    def _expression_from(self, args: Mapping[str, Any]) -> str:
        """Accept the expression, or rebuild it from an operation triple.

        Models ignore the declared arg shape: Qwen3-4B sent
        ``{"operation": "multiply", "a": 199.0, "b": 7.5}`` and every one of its
        eight steps failed on a schema complaint instead of learning the task.
        Rebuilding the expression keeps the episode about the arithmetic.
        """

        for key in ("expression", "code", "expr"):
            value = args.get(key)
            if isinstance(value, str) and value.strip():
                return value

        op = str(args.get("operation") or args.get("op") or "").lower()
        symbol = self._OPS.get(op)
        left, right = args.get("a"), args.get("b")
        if symbol and left is not None and right is not None:
            try:
                return f"({float(left)}){symbol}({float(right)})"
            except (TypeError, ValueError):
                pass
        raise ValueError(
            "calc needs {'expression': '<expr>'}, or operation/a/b "
            "(for example {'operation': 'multiply', 'a': 199.0, 'b': 7.5})"
        )

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        if name == "calc":
            return self._calc.run(self._expression_from(args))
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
    out.extend(_aggregate_tasks(ids))
    return out


def _aggregate_tasks(ids: tuple[str, ...]) -> list[LookupTask]:
    """Tasks that need every order fetched before anything can be computed.

    The two-call tasks above turned out to be free once the tool schemas were
    spelled out in the prompt: 74 of 75 solved 4/4. What is actually hard for a
    4B model is holding six numbers from six separate observations in context and
    combining them, which is what these require.
    """

    amounts = [_amount(oid) for oid in ids]
    listed = "、".join(ids)
    literal = "[" + ", ".join(repr(a) for a in amounts) + "]"
    out: list[LookupTask] = []

    out.append(
        _build(
            "agg-sum-all",
            f"订单 {listed} 的金额合计是多少元？需要逐个查询。",
            _fmt(round(sum(amounts), 2)),
            ids,
            f"round(sum({literal}), 2)",
        )
    )
    out.append(
        _build(
            "agg-avg-all",
            f"订单 {listed} 的平均金额是多少元？（保留两位小数）",
            _fmt(round(sum(amounts) / len(amounts), 2)),
            ids,
            f"round(sum({literal})/len({literal}), 2)",
        )
    )
    out.append(
        _build(
            "agg-max-all",
            f"订单 {listed} 中金额最大的是多少元？",
            _fmt(round(max(amounts), 2)),
            ids,
            f"round(max({literal}), 2)",
        )
    )
    out.append(
        _build(
            "agg-min-all",
            f"订单 {listed} 中金额最小的是多少元？",
            _fmt(round(min(amounts), 2)),
            ids,
            f"round(min({literal}), 2)",
        )
    )

    threshold = 200
    above = [a for a in amounts if a > threshold]
    out.append(
        _build(
            f"agg-count-over-{threshold}",
            f"订单 {listed} 中金额超过 {threshold} 元的有几个？",
            str(len(above)),
            ids,
            f"len([x for x in {literal} if x > {threshold}])",
        )
    )

    # The measured weak spot: summing a literal list works, but logic *over* the
    # list (max, min, count, median, filtered sums) fails 3 times out of 3. These
    # variants all need a comparison the model cannot read off the observations.
    mean = sum(amounts) / len(amounts)
    out.append(
        _build(
            "agg-count-over-mean",
            f"订单 {listed} 中，金额高于平均值的有几个？",
            str(len([a for a in amounts if a > mean])),
            ids,
            f"len([x for x in {literal} if x > sum({literal})/len({literal})])",
        )
    )
    out.append(
        _build(
            "agg-max-minus-min",
            f"订单 {listed} 中最大金额与最小金额相差多少元？",
            _fmt(round(max(amounts) - min(amounts), 2)),
            ids,
            f"round(max({literal}) - min({literal}), 2)",
        )
    )
    out.append(
        _build(
            "agg-sum-over-500",
            f"订单 {listed} 中金额超过 500 元的那些订单，金额合计是多少元？",
            _fmt(round(sum(a for a in amounts if a > 500), 2)),
            ids,
            f"round(sum([x for x in {literal} if x > 500]), 2)",
        )
    )
    out.append(
        _build(
            "agg-median",
            f"订单 {listed} 的金额中位数是多少元？（保留两位小数）",
            _fmt(round(sorted(amounts)[len(amounts) // 2 - 1] if len(amounts) % 2 == 0 else sorted(amounts)[len(amounts) // 2], 2)),
            ids,
            f"round(median({literal}), 2)",
        )
    )
    out.append(
        _build(
            "agg-sum-squares",
            f"订单 {listed} 各自金额的平方和是多少？（保留两位小数）",
            _fmt(round(sum(a * a for a in amounts), 2)),
            ids,
            f"round(sum([x*x for x in {literal}]), 2)",
        )
    )

    out.extend(_subset_tasks(ids))
    return out


def _subset_tasks(ids: tuple[str, ...]) -> list[LookupTask]:
    """Two-, three- and four-order subsets with logic operations over the values.

    The six-order versions above only yield ten tasks, and ten is too thin to see
    a training curve. Subsets keep the same failure mode (logic over several
    observed values) while giving the sampler enough variety. Sizes are mixed on
    purpose: three values might be tractable while four are not, and that boundary
    is exactly where a learnable pass rate lives.
    """

    from itertools import combinations

    out: list[LookupTask] = []

    def add(task_id: str, instruction: str, answer: str, subset, expression: str) -> None:
        out.append(_build(task_id, instruction, answer, subset, expression))

    for size in (2, 3, 4):
        for subset in combinations(ids, size):
            amounts = [_amount(oid) for oid in subset]
            literal = "[" + ", ".join(repr(a) for a in amounts) + "]"
            listed = "、".join(subset)
            tag = "-".join(o[-3:] for o in subset)

            add(
                f"sub-max-{tag}",
                f"订单 {listed} 中金额最大的是多少元？",
                _fmt(round(max(amounts), 2)),
                subset,
                f"round(max({literal}), 2)",
            )
            add(
                f"sub-min-{tag}",
                f"订单 {listed} 中金额最小的是多少元？",
                _fmt(round(min(amounts), 2)),
                subset,
                f"round(min({literal}), 2)",
            )
            if size >= 3:
                add(
                    f"sub-count-mean-{tag}",
                    f"订单 {listed} 中，金额高于这几个订单平均值的有几个？",
                    str(len([a for a in amounts if a > sum(amounts) / len(amounts)])),
                    subset,
                    f"len([x for x in {literal} if x > sum({literal})/len({literal})])",
                )
                add(
                    f"sub-over-{tag}",
                    f"订单 {listed} 中金额超过 500 元的那部分合计多少元？没有则填 0。",
                    _fmt(round(sum(a for a in amounts if a > 500), 2)),
                    subset,
                    f"round(sum([x for x in {literal} if x > 500]), 2)",
                )
            if size == 4:
                add(
                    f"sub-median-{tag}",
                    f"订单 {listed} 的金额中位数是多少元？（保留两位小数）",
                    _fmt(round((sorted(amounts)[1] + sorted(amounts)[2]) / 2, 2)),
                    subset,
                    f"round(median({literal}), 2)",
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
