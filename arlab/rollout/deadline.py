"""Partial rollout: stop waiting for the slowest episode in a group.

Why this belongs in agentic RL and not in general training infra: a single rollout
here is a *sequence* of tool calls, and the sequence length varies by an order of
magnitude across samples of the same task (one episode finishes in a call, the next
needs nine). A synchronous group waits for the last one, so most of the batch sits
idle for most of the step. Measured on this repo's box, eight rollouts of one task
took 40.8s issued one at a time and 6.1s issued together -- and even then the group
cannot finish faster than its slowest member.

The contract with ``run_task`` is what makes truncation cheap: when the policy
returns ``None`` the runner stops the episode, scores whatever trajectory exists,
and returns a normal record. So a straggler does not need to be cancelled or
killed; it just stops being extended.

What it costs: a truncated episode is scored on a partial trajectory, which is
usually a bad answer. That is the trade -- bounded step latency in exchange for a
lower reward on slow samples. ``collect_with_deadline`` reports how many were cut
so the caller can decide whether the trade was worth it.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from arlab.rollout.runner import RunnerConfig, run_task


@dataclass
class RolloutStats:
    """What the group actually cost."""

    n: int = 0
    wall_seconds: float = 0.0
    truncated: int = 0
    solved: int = 0
    durations: list[float] = field(default_factory=list)

    @property
    def waste_fraction(self) -> float:
        return self.truncated / self.n if self.n else 0.0

    def percentile(self, q: float) -> float:
        if not self.durations:
            return 0.0
        ordered = sorted(self.durations)
        index = min(len(ordered) - 1, max(0, int(len(ordered) * q) - 1))
        return ordered[index]

    def summary(self) -> str:
        return (
            f"n={self.n} wall={self.wall_seconds:.1f}s solved={self.solved} "
            f"truncated={self.truncated} p50={self.percentile(0.5):.1f}s "
            f"p99={self.percentile(0.99):.1f}s"
        )


class DeadlinePolicy:
    """Wraps a policy and returns ``None`` once the episode has run too long.

    ``None`` is the runner's "stop this episode" signal, so the episode is
    finalised with the steps it already has and scored like any other.
    """

    def __init__(self, inner: Callable[[Any, Sequence[Any]], Any], deadline_s: float | None) -> None:
        self.inner = inner
        self.deadline_s = deadline_s
        self.started = time.perf_counter()
        self.truncated = False

    def __call__(self, task: Any, steps: Sequence[Any]) -> Any:
        if self.deadline_s is not None and time.perf_counter() - self.started > self.deadline_s:
            self.truncated = True
            return None
        return self.inner(task, steps)


def collect_with_deadline(
    task: Any,
    env_factory: Callable[[], Any],
    policy_factory: Callable[[int], Any],
    cfg: RunnerConfig,
    *,
    group_size: int,
    workers: int = 8,
    deadline_s: float | None = None,
    policy_name: str = "llm",
) -> tuple[list[Any], RolloutStats]:
    """Run one group concurrently, truncating any episode that outlives ``deadline_s``."""

    stats = RolloutStats(n=group_size)
    wrapped: dict[int, DeadlinePolicy] = {}
    results: dict[int, Any] = {}
    lock = threading.Lock()
    started = time.perf_counter()

    def one(index: int) -> None:
        seed = cfg.seed * 100 + index
        policy = DeadlinePolicy(policy_factory(seed), deadline_s)
        with lock:
            wrapped[index] = policy
        record = run_task(
            task,
            env_factory,
            policy,
            dataclasses.replace(cfg, seed=seed),
            policy_name=policy_name,
            store=None,
        )
        with lock:
            results[index] = record

    if workers <= 1:
        for index in range(group_size):
            one(index)
    else:
        with ThreadPoolExecutor(max_workers=min(workers, group_size)) as pool:
            futures = [pool.submit(one, index) for index in range(group_size)]
            for future in as_completed(futures):
                future.result()

    stats.wall_seconds = time.perf_counter() - started
    for index in range(group_size):
        record = results.get(index)
        if record is None:
            continue
        stats.durations.append(float(getattr(record, "meta", {}).get("wall_seconds", 0.0)))
        if record.reward.get("r_task", 0.0) >= 0.999:
            stats.solved += 1
        if wrapped.get(index) is not None and wrapped[index].truncated:
            stats.truncated += 1

    return [results[index] for index in range(group_size) if index in results], stats
