"""Turn a baseline run into a smaller, learnable task set.

GRPO learns from *variance inside a group*. A task the model always solves, or
never solves, produces identical rewards across its samples, so its advantage is
zero and it contributes nothing but compute. This module does the arithmetic:
read the baseline traces, compute each task's pass rate, keep the band.

"Pass" means ``r_task == 1.0`` -- not "reward > 0". With partial credit almost
every rollout has a positive total, so the looser definition reports everything
as solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from arlab.store.trace import TraceRecord
from arlab.tasks.stage_io import LoadedTask


PASS_THRESHOLD = 1.0


@dataclass(frozen=True)
class TaskStats:
    task_id: str
    n: int
    pass_rate: float
    mean_reward: float


@dataclass
class FilterResult:
    kept: list[LoadedTask]
    too_easy: list[LoadedTask]
    too_hard: list[LoadedTask]
    unmeasured: list[LoadedTask]
    stats: dict[str, TaskStats]

    @property
    def dropped(self) -> int:
        return len(self.too_easy) + len(self.too_hard)

    def summary(self) -> str:
        return (
            f"kept={len(self.kept)} too_easy={len(self.too_easy)} "
            f"too_hard={len(self.too_hard)} unmeasured={len(self.unmeasured)}"
        )


def pass_rates(
    records: Iterable[TraceRecord],
    config_id: str | None = None,
) -> dict[str, TaskStats]:
    """Per-task pass rate over the rollouts in the store."""

    buckets: dict[str, list[float]] = {}
    for record in records:
        if config_id is not None and record.config_id != config_id:
            continue
        solved = 1.0 if float(record.reward.get("r_task", 0.0)) >= PASS_THRESHOLD else 0.0
        buckets.setdefault(record.task_id, []).append(solved)

    stats: dict[str, TaskStats] = {}
    for task_id, solved in buckets.items():
        n = len(solved)
        stats[task_id] = TaskStats(
            task_id=task_id,
            n=n,
            pass_rate=sum(solved) / n if n else 0.0,
            mean_reward=sum(solved) / n if n else 0.0,
        )
    return stats


def filter_tasks(
    tasks: Sequence[LoadedTask],
    stats: Mapping[str, TaskStats],
    lo: float = 0.2,
    hi: float = 0.8,
    keep_unmeasured: bool = True,
) -> FilterResult:
    """Split tasks into keep / too easy / too hard / unmeasured."""

    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError("need 0 <= lo < hi <= 1")

    kept: list[LoadedTask] = []
    too_easy: list[LoadedTask] = []
    too_hard: list[LoadedTask] = []
    unmeasured: list[LoadedTask] = []

    for loaded in tasks:
        stat = stats.get(loaded.task.task_id)
        if stat is None:
            unmeasured.append(loaded)
            if keep_unmeasured:
                kept.append(loaded)
            continue
        if stat.pass_rate > hi:
            too_easy.append(loaded)
        elif stat.pass_rate < lo:
            too_hard.append(loaded)
        else:
            kept.append(loaded)

    return FilterResult(
        kept=kept,
        too_easy=too_easy,
        too_hard=too_hard,
        unmeasured=unmeasured,
        stats=dict(stats),
    )
