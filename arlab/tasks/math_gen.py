"""Deterministic math task generator, three difficulty tiers.

Why this exists: six tasks cannot train a GRPO policy. With ``group_size=8`` a
4B model very likely solves all six, every group gets identical reward, the
advantage is zero, and nothing is learned. A learnable set needs enough tasks
that a meaningful fraction lands in the 20-80% pass band.

Every generated task carries the expression that produces its answer, and
``verify()`` executes it with the same restricted interpreter the agent uses. So
a template bug surfaces here, offline, instead of as a mysterious reward.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from arlab.tasks.math_env import ArithmeticExecutor

TIERS = ("easy", "medium", "hard")


@dataclass(frozen=True)
class GeneratedTask:
    task_id: str
    instruction: str
    code: str
    answer: str
    tier: str


def _fmt(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _easy(rng: random.Random) -> tuple[str, str]:
    a, b, c = rng.randint(12, 99), rng.randint(3, 19), rng.randint(1, 50)
    if rng.random() < 0.5:
        return f"{a} × {b} + {c} 等于多少？", f"{a}*{b}+{c}"
    a = rng.randint(11, 40)
    b = rng.randint(2, 15)
    return f"{a} 的平方减去 {b} 等于多少？", f"{a}**2-{b}"


def _medium(rng: random.Random) -> tuple[str, str]:
    """Two-hop word problems: the model must derive a quantity, not copy an expression.

    A single arithmetic prompt is solved by writing the expression straight down,
    which a 4B model does reliably -- 48/48 at baseline, i.e. zero advantage for
    GRPO. Requiring an intermediate quantity is what creates variance.
    """

    kind = rng.randrange(3)
    if kind == 0:
        price = rng.randrange(60, 300, 10)
        count = rng.randint(3, 12)
        off = rng.choice([7.5, 8.0, 8.5, 9.0])
        ship = rng.randint(5, 30)
        return (
            f"某商品单价 {price} 元，买 {count} 件，按总价 {off}/10 付款，另付运费 {ship} 元。"
            f"实际支付多少元？",
            f"round({price}*{count}*{off}/10 + {ship}, 2)",
        )
    if kind == 1:
        base = rng.randrange(200, 2000, 50)
        first = rng.choice([5, 8, 10, 12, 15])
        second = rng.choice([3, 6, 9, 11])
        return (
            f"某厂去年产量 {base} 件，今年比去年增长 {first}%，明年预计再增长 {second}%。"
            f"明年产量预计多少件？（保留两位小数）",
            f"round({base}*(1+{first}/100)*(1+{second}/100), 2)",
        )
    distance = rng.randint(120, 900)
    speed = rng.choice([40, 50, 60, 75, 80, 90])
    rest = rng.choice([15, 20, 30, 45])
    return (
        f"从甲地到乙地 {distance} 公里，车速 {speed} 公里/小时，途中休息 {rest} 分钟。"
        f"总耗时多少小时？（保留两位小数）",
        f"round({distance}/{speed} + {rest}/60, 2)",
    )


def _hard(rng: random.Random) -> tuple[str, str]:
    """Three-hop word problems: several intermediate quantities, no given formula."""

    kind = rng.randrange(3)
    if kind == 0:
        a = rng.randrange(500, 5000, 100)
        pct = rng.choice([10, 20, 25, 40, 50])
        return (
            f"甲有 {a} 元，乙比甲多 {pct}%，丙是甲乙两人之和的一半。三人一共有多少元？"
            f"（保留两位小数）",
            f"round({a} + {a}*(1+{pct}/100) + ({a} + {a}*(1+{pct}/100))/2, 2)",
        )
    if kind == 1:
        cost = rng.randrange(80, 900, 10)
        markup = rng.choice([20, 30, 40, 50])
        off = rng.choice([7.5, 8.0, 8.5, 9.0])
        return (
            f"某商品成本 {cost} 元，加价 {markup}% 作为标价，再按标价的 {off}/10 售出。"
            f"利润率是多少百分比？（保留两位小数）",
            f"round((({cost}*(1+{markup}/100)*{off}/10) - {cost})/{cost}*100, 2)",
        )
    mass_a = rng.randint(10, 60)
    mass_b = rng.randint(10, 60)
    pct_a = rng.choice([5, 10, 15, 20])
    pct_b = rng.choice([25, 30, 40, 50])
    return (
        f"把 {mass_a} 千克浓度 {pct_a}% 的盐水和 {mass_b} 千克浓度 {pct_b}% 的盐水混合，"
        f"混合后浓度是多少百分比？（保留两位小数）",
        f"round(({mass_a}*{pct_a}/100 + {mass_b}*{pct_b}/100)/({mass_a}+{mass_b})*100, 2)",
    )


BUILDERS: dict[str, Callable[[random.Random], tuple[str, str]]] = {
    "easy": _easy,
    "medium": _medium,
    "hard": _hard,
}


def verify(tasks: list[GeneratedTask], executor: ArithmeticExecutor | None = None) -> list[str]:
    """Execute each expression; return a list of problems (empty means all good)."""

    runner = executor or ArithmeticExecutor()
    problems: list[str] = []
    for task in tasks:
        try:
            got = runner.run(task.code)
        except Exception as exc:  # noqa: BLE001 - report, never raise
            problems.append(f"{task.task_id}: expression failed ({type(exc).__name__}: {exc})")
            continue
        if _fmt(float(got)) != _fmt(float(task.answer)):
            problems.append(f"{task.task_id}: expression gives {got}, task says {task.answer}")
    return problems


def generate(per_tier: int = 16, seed: int = 0) -> list[GeneratedTask]:
    """``per_tier`` tasks for each tier, ids stable for a given seed."""

    if per_tier <= 0:
        raise ValueError("per_tier must be positive")
    tasks: list[GeneratedTask] = []
    for tier in TIERS:
        rng = random.Random(f"{seed}-{tier}")
        seen: set[str] = set()
        index = 0
        while sum(1 for t in tasks if t.tier == tier) < per_tier:
            instruction, code = BUILDERS[tier](rng)
            if instruction in seen:
                continue
            seen.add(instruction)
            index += 1
            value = ArithmeticExecutor().run(code)
            tasks.append(
                GeneratedTask(
                    task_id=f"gen-{tier}-{index:02d}",
                    instruction=instruction + " 用 python 算，然后提交答案。",
                    code=code,
                    answer=_fmt(value),
                    tier=tier,
                )
            )
    return tasks


def tier_counts(tasks: list[GeneratedTask]) -> dict[str, int]:
    counts = {tier: 0 for tier in TIERS}
    for task in tasks:
        counts[task.tier] = counts.get(task.tier, 0) + 1
    return counts
