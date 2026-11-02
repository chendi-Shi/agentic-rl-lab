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
    kind = rng.randrange(3)
    if kind == 0:
        price = rng.randrange(120, 960, 20)
        off = rng.choice([7.5, 8.0, 8.5, 9.0])
        return (
            f"商品原价 {price} 元，打 {off} 折，现价是多少元？",
            f"{price}*{off}/10",
        )
    if kind == 1:
        nums = [rng.randint(10, 90) for _ in range(3)]
        return (
            f"三个数 {nums[0]}、{nums[1]}、{nums[2]} 的平均值是多少？",
            f"({nums[0]}+{nums[1]}+{nums[2]})/3",
        )
    a = rng.choice([3, 4, 5, 6])
    factor = rng.randint(4, 12)
    b = a * factor
    c = rng.randint(7, 30)
    return (
        f"若 {a}:{b} = {c}:x，求 x 的值。",
        f"{b}*{c}/{a}",
    )


def _hard(rng: random.Random) -> tuple[str, str]:
    kind = rng.randrange(3)
    if kind == 0:
        # Build the equation from two integer roots, so the discriminant is always
        # a non-negative perfect square (a random p, q pair can go complex).
        r1 = rng.randint(1, 9)
        r2 = rng.randint(2, 9)
        if r1 == r2:
            r2 += 1
        p, q = r1 + r2, r1 * r2
        return (
            f"方程 x² - {p}x + {q} = 0 的较大根是多少？（保留两位小数）",
            f"round(({p}+({p}**2-4*{q})**0.5)/2, 2)",
        )
    if kind == 1:
        r = rng.randint(2, 15)
        return (
            f"半径为 {r} 的圆面积是多少？（π 取 3.14，保留两位小数）",
            f"round(3.14*{r}**2, 2)",
        )
    principal = rng.randrange(1000, 9000, 500)
    rate = rng.choice([3, 4, 5])
    years = rng.randint(2, 4)
    return (
        f"本金 {principal} 元，年利率 {rate}%，复利 {years} 年后本息合计多少？（保留两位小数）",
        f"round({principal}*(1+{rate}/100)**{years}, 2)",
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
