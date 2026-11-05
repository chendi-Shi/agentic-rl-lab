"""Run one task's rollouts concurrently.

The training loop used to call ``run_task`` in a for-loop, waiting for one episode
to finish before starting the next. On this box that left 85% of the achievable
throughput on the table: eight rollouts of the same task took 40.8s sequentially
and 6.1s when issued together, with identical rewards (scripts/bench_rollout.py).

Concurrency is safe here because every rollout gets its own environment instance
(``run_task`` calls ``env_factory()``) and its own ``ChatClient`` (``policy_factory``).
The trace store is *not* used in the concurrent path: ``TraceStore`` appends to a
single file and is not thread-safe, so callers that need resumability should run
with ``workers=1``.
"""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Sequence

from arlab.rollout.runner import RunnerConfig, run_task


def collect_records(
    task: Any,
    env_factory: Callable[[], Any],
    policy_factory: Callable[[int], Any],
    cfg: RunnerConfig,
    *,
    group_size: int,
    workers: int = 8,
    policy_name: str = "llm",
) -> list[Any]:
    """One record per rollout, returned in sample order regardless of finish order."""

    if workers <= 1:
        return [
            run_task(
                task,
                env_factory,
                policy_factory(cfg.seed * 100 + index),
                dataclasses.replace(cfg, seed=cfg.seed * 100 + index),
                policy_name=policy_name,
                store=None,
            )
            for index in range(group_size)
        ]

    results: dict[int, Any] = {}
    lock = threading.Lock()

    def one(index: int) -> None:
        seed = cfg.seed * 100 + index
        record = run_task(
            task,
            env_factory,
            policy_factory(seed),
            dataclasses.replace(cfg, seed=seed),
            policy_name=policy_name,
            store=None,
        )
        with lock:
            results[index] = record

    with ThreadPoolExecutor(max_workers=min(workers, group_size)) as pool:
        futures = [pool.submit(one, index) for index in range(group_size)]
        for future in as_completed(futures):
            future.result()

    return [results[index] for index in range(group_size)]


def group_latency(records: Sequence[Any]) -> float:
    """Slowest episode in the group: the number partial rollout is trying to cut."""

    values = [getattr(record, "wall_seconds", 0.0) for record in records]
    return max(values) if values else 0.0
