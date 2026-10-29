"""Offline scheduler model: quantify the long-tail loss before renting a card.

Two infra problems in agentic RL, both caused by episode length being
heavy-tailed and *unknown in advance*:

1. **Slot under-utilisation.** A fixed batch advances at its slowest member, so
   every other slot idles from its own finish until the tail finishes. Refilling
   slots as they free (async, as in AReaL / verl's partial rollout) recovers it.
2. **Round-budget truncation.** A training round has a fixed step budget. Without
   partial rollout, an episode longer than the budget is cut and scored as a
   failure -- all its steps become wasted compute and, worse, the reward is
   biased against long-horizon tasks.

This is a closed-loop model over lengths alone, not a profiler: it answers "how
much is on the table", needs no GPU, and later takes measured per-step latencies
so the comparison stays valid.

All times are milliseconds; ``step_time_ms`` is the cost of one agent step
(model call + tool call) for one episode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SimResult:
    mode: str
    n_episodes: int
    concurrency: int
    wall_ms: float
    total_step_ms: float
    slot_utilization: float
    idle_slot_ms: float
    mean_completion_ms: float
    tail_share: float

    def summary(self) -> str:
        return (
            f"{self.mode:<16} wall={self.wall_ms / 1000:7.2f}s  "
            f"util={self.slot_utilization:5.1%}  idle={self.idle_slot_ms / 1000:7.2f}s  "
            f"mean_lat={self.mean_completion_ms / 1000:6.2f}s"
        )


def _check(lengths: Sequence[int], concurrency: int) -> None:
    if not lengths:
        raise ValueError("no episodes")
    if any(length <= 0 for length in lengths):
        raise ValueError("episode length must be positive")
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")


def tail_share(lengths: Sequence[int], frac: float = 0.05) -> float:
    """Share of total steps contributed by the longest ``frac`` of episodes."""

    total = sum(lengths)
    if total == 0:
        return 0.0
    k = max(1, int(len(lengths) * frac))
    return sum(sorted(lengths)[-k:]) / total


def simulate_sync_batch(
    lengths: Sequence[int], concurrency: int, step_time_ms: float = 1.0
) -> SimResult:
    """Fixed batches of ``concurrency``; the batch advances at its slowest member."""

    _check(lengths, concurrency)
    wall = 0.0
    total = 0.0
    idle = 0.0
    completions: list[float] = []

    for start in range(0, len(lengths), concurrency):
        batch = list(lengths[start : start + concurrency])
        longest = max(batch)
        batch_wall = longest * step_time_ms
        wall += batch_wall
        for length in batch:
            total += length * step_time_ms
            idle += (longest - length) * step_time_ms
            completions.append(wall)

    return SimResult(
        mode="sync_batch",
        n_episodes=len(lengths),
        concurrency=concurrency,
        wall_ms=wall,
        total_step_ms=total,
        slot_utilization=total / (wall * concurrency) if wall else 0.0,
        idle_slot_ms=idle,
        mean_completion_ms=sum(completions) / len(completions),
        tail_share=tail_share(lengths),
    )


def simulate_async_refill(
    lengths: Sequence[int], concurrency: int, step_time_ms: float = 1.0
) -> SimResult:
    """Keep all slots busy: each episode starts at the earliest free slot."""

    _check(lengths, concurrency)
    slot_free = [0.0] * concurrency
    completions: list[float] = []
    total = 0.0

    for length in lengths:
        index = min(range(concurrency), key=lambda i: slot_free[i])
        service = length * step_time_ms
        finish = slot_free[index] + service
        slot_free[index] = finish
        total += service
        completions.append(finish)

    wall = max(slot_free)
    return SimResult(
        mode="async_refill",
        n_episodes=len(lengths),
        concurrency=concurrency,
        wall_ms=wall,
        total_step_ms=total,
        slot_utilization=total / (wall * concurrency) if wall else 0.0,
        idle_slot_ms=wall * concurrency - total,
        mean_completion_ms=sum(completions) / len(completions),
        tail_share=tail_share(lengths),
    )


@dataclass(frozen=True)
class TruncationResult:
    mode: str
    round_budget: int
    n_episodes: int
    finished: int
    truncated: int
    executed_steps: int
    wasted_steps: int

    @property
    def waste_fraction(self) -> float:
        return self.wasted_steps / self.executed_steps if self.executed_steps else 0.0

    def summary(self) -> str:
        return (
            f"{self.mode:<16} finished={self.finished:4d}  truncated={self.truncated:4d}  "
            f"steps={self.executed_steps:6d}  wasted={self.wasted_steps:5d} "
            f"({self.waste_fraction:5.1%})"
        )


def simulate_truncation(
    lengths: Sequence[int], round_budget: int, partial_rollout: bool
) -> TruncationResult:
    """Cost of carrying episodes across training rounds.

    ``partial_rollout=False``: an episode longer than the round budget is cut at
    the budget and scored as a failure. Its steps are executed but yield no
    successful signal, and the reward becomes biased against long-horizon tasks.

    ``partial_rollout=True``: the episode keeps its state and resumes next round.
    Every step is executed exactly once and no episode is lost.
    """

    _check(lengths, round_budget)
    executed = 0
    finished = 0
    truncated = 0
    wasted = 0

    for length in lengths:
        if partial_rollout or length <= round_budget:
            executed += length
            finished += 1
            continue
        executed += round_budget
        truncated += 1
        wasted += round_budget

    return TruncationResult(
        mode="partial_rollout" if partial_rollout else "truncate",
        round_budget=round_budget,
        n_episodes=len(lengths),
        finished=finished,
        truncated=truncated,
        executed_steps=executed,
        wasted_steps=wasted,
    )


def compare(
    lengths: Sequence[int], concurrency: int, step_time_ms: float = 1.0
) -> dict[str, SimResult]:
    return {
        "sync_batch": simulate_sync_batch(lengths, concurrency, step_time_ms),
        "async_refill": simulate_async_refill(lengths, concurrency, step_time_ms),
    }


def pareto_lengths(n: int, shape: float = 1.5, scale: int = 3, cap: int = 40, seed: int = 0) -> list[int]:
    """Heavy-tailed episode lengths, clipped to a plausible agent range."""

    rng = random.Random(seed)
    out: list[int] = []
    while len(out) < n:
        value = int(scale / (rng.random() ** (1.0 / shape)))
        if value >= 1:
            out.append(min(value, cap))
    return out
