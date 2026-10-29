"""Stability diagnostics for GRPO groups.

GRPO computes the advantage within a group of samples for the same task. If
every sample in the group gets the same reward, the group's std is zero, the
normalised advantage is zero, and **no gradient flows** -- the run looks healthy
(loss moves, steps tick) while learning nothing.

That is the single most common way agentic RL silently fails, so it gets a
first-class diagnostic instead of a log line.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class GroupStats:
    task_id: str
    n: int
    mean: float
    std: float
    min: float
    max: float
    pass_rate: float


@dataclass
class CollapseReport:
    n_groups: int
    n_collapsed: int
    frac_collapsed: float
    collapsed_task_ids: list[str] = field(default_factory=list)
    healthy: bool = True

    def summary(self) -> str:
        if self.n_groups == 0:
            return (
                "[NO DATA] every task had fewer than 2 samples; GRPO needs a group "
                "(group_size >= 2) before any advantage exists"
            )
        verdict = "OK" if self.healthy else "COLLAPSED"
        return (
            f"[{verdict}] {self.n_collapsed}/{self.n_groups} groups have zero reward variance "
            f"({self.frac_collapsed:.1%})"
        )


@dataclass
class DifficultyReport:
    kept: list[str] = field(default_factory=list)
    too_easy: list[str] = field(default_factory=list)
    too_hard: list[str] = field(default_factory=list)
    stats: list[GroupStats] = field(default_factory=list)


def group_stats(
    rewards_by_task: dict[str, Sequence[float]],
    pass_threshold: float = 1.0,
) -> list[GroupStats]:
    """Per-task mean/std/pass-rate over the sampled rollouts.

    ``pass_threshold`` defaults to 1.0 because partial credit means almost every
    rollout has a positive total. Counting "solved" as ``total > 0`` would report
    a 100% pass rate for a task the model never actually completes.
    """

    out: list[GroupStats] = []
    for task_id, rewards in rewards_by_task.items():
        if not rewards:
            continue
        arr = np.asarray(list(rewards), dtype=float)
        out.append(
            GroupStats(
                task_id=task_id,
                n=int(arr.size),
                mean=float(arr.mean()),
                std=float(arr.std()),
                min=float(arr.min()),
                max=float(arr.max()),
                pass_rate=float((arr >= pass_threshold).mean()),
            )
        )
    return out


def detect_advantage_collapse(
    rewards_by_task: dict[str, Sequence[float]],
    atol: float = 1e-8,
    max_frac: float = 0.2,
) -> CollapseReport:
    """Flag groups whose reward variance is effectively zero.

    ``max_frac`` is the tolerance for collapsed groups before the batch is
    declared unhealthy.
    """

    collapsed: list[str] = []
    n_groups = 0
    for task_id, rewards in rewards_by_task.items():
        if len(rewards) < 2:
            continue
        n_groups += 1
        arr = np.asarray(list(rewards), dtype=float)
        if float(arr.std()) <= atol:
            collapsed.append(task_id)

    frac = (len(collapsed) / n_groups) if n_groups else 1.0
    return CollapseReport(
        n_groups=n_groups,
        n_collapsed=len(collapsed),
        frac_collapsed=frac,
        collapsed_task_ids=sorted(collapsed),
        healthy=(n_groups > 0 and frac <= max_frac),
    )


def classify_difficulty(
    rewards_by_task: dict[str, Sequence[float]],
    lo: float = 0.2,
    hi: float = 0.8,
    pass_threshold: float = 1.0,
) -> DifficultyReport:
    """Keep tasks whose pass rate lands in the learnable band.

    Uses ``reward >= pass_threshold`` as the pass definition, matching a fully
    solved task once penalties are included.
    """

    report = DifficultyReport()
    for stat in group_stats(rewards_by_task, pass_threshold=pass_threshold):
        report.stats.append(stat)
        if stat.pass_rate < lo:
            report.too_hard.append(stat.task_id)
        elif stat.pass_rate > hi:
            report.too_easy.append(stat.task_id)
        else:
            report.kept.append(stat.task_id)
    return report


def length_percentiles(lengths: Iterable[int]) -> dict[str, float]:
    """Trajectory-length distribution; the p99 drives batch wall-clock."""

    arr = np.asarray(list(lengths), dtype=float)
    if arr.size == 0:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "tail_share": 0.0}
    p50, p90, p99 = np.percentile(arr, [50, 90, 99])
    # Share of total steps contributed by the slowest 5% of trajectories.
    # Take the top-k by value rather than by threshold: when most trajectories
    # share the same length, a percentile threshold selects everyone.
    k = max(1, int(np.ceil(0.05 * arr.size)))
    tail = np.sort(arr)[-k:]
    tail_share = float(tail.sum() / arr.sum()) if arr.sum() > 0 else 0.0
    return {
        "p50": float(p50),
        "p90": float(p90),
        "p99": float(p99),
        "mean": float(arr.mean()),
        "tail_share": tail_share,
    }
