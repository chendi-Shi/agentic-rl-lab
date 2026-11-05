"""Measure what concurrency and tail truncation actually buy on this box.

Three modes over the same task and the same number of rollouts:

  A sequential      what the training loop does today (one rollout at a time)
  B concurrent      all rollouts in flight at once; vLLM batches them
  C concurrent+cut  same, but an episode that runs past a per-episode deadline is
                    truncated (the policy returns None, the runner finalises what
                    it has) instead of holding the whole group hostage

The point is not that B is faster -- it obviously is. The point is to measure how
much, and what truncation costs in solved count. That trade is the whole argument
for partial rollout in agentic RL, where one 9-step episode next to seven 1-step
ones wastes most of the batch.
"""

from __future__ import annotations

import dataclasses
import json
import statistics as st
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from arlab.policy.openai import ChatClient, LLMPolicy, system_prompt
from arlab.rollout.runner import RunnerConfig, run_task
from arlab.tasks.lookup_tasks import LookupEnv

BASE_URL = "http://127.0.0.1:6006"
MODEL = "qwen3-4b"
GROUP = 8


def make_policy(seed: int, deadline_s: float | None = None):
    client = ChatClient(
        base_url=BASE_URL,
        model=MODEL,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "seed": seed},
    )
    llm = LLMPolicy(client, max_tokens=512, temperature=0.9)
    started = time.perf_counter()

    def policy(task, steps):
        if deadline_s is not None and time.perf_counter() - started > deadline_s:
            return None  # truncate: the runner finalises the episode as-is
        return llm(task, steps)

    return policy


def timed(record):
    return record.wall_seconds if hasattr(record, "wall_seconds") else 0.0


def run_mode(name, task, cfg, *, concurrent: bool, deadline_s: float | None):
    started = time.perf_counter()
    records = []

    if concurrent:
        lock = threading.Lock()

        def one(index):
            seed = cfg.seed * 100 + index
            policy = make_policy(seed, deadline_s)
            record = run_task(
                task,
                LookupEnv,
                policy,
                dataclasses.replace(cfg, seed=seed),
                policy_name="bench",
                store=None,
            )
            with lock:
                records.append(record)
            return record

        with ThreadPoolExecutor(max_workers=GROUP) as pool:
            futures = [pool.submit(one, i) for i in range(GROUP)]
            for future in as_completed(futures):
                future.result()
    else:
        for index in range(GROUP):
            seed = cfg.seed * 100 + index
            records.append(
                run_task(
                    task,
                    LookupEnv,
                    make_policy(seed, deadline_s),
                    dataclasses.replace(cfg, seed=seed),
                    policy_name="bench",
                    store=None,
                )
            )

    wall = time.perf_counter() - started
    solved = sum(1 for r in records if r.reward["r_task"] >= 0.999)
    durations = sorted(timed(r) for r in records)
    p50 = st.median(durations) if durations else 0.0
    p99 = durations[int(len(durations) * 0.99) - 1] if durations else 0.0
    print(
        f"{name:<16} wall={wall:6.1f}s  solved={solved}/{GROUP}  "
        f"p50={p50:5.1f}s  max={max(durations) if durations else 0:5.1f}s  "
        f"throughput={GROUP/wall*60:4.1f} rollouts/min"
    )
    return {
        "mode": name,
        "wall_s": round(wall, 1),
        "solved": solved,
        "p50_s": round(p50, 1),
        "max_s": round(max(durations) if durations else 0.0, 1),
        "throughput_per_min": round(GROUP / wall * 60, 1),
    }


def main() -> int:
    tasks = json.loads(Path("tasks/stage3-hard.json").read_text(encoding="utf-8"))
    task = next(t for t in tasks if t["task_id"] == "agg-sum-all")
    from arlab.tasks.stage_io import Task as _  # noqa: F401 - keep import graph honest

    built = None
    for candidate in tasks:
        if candidate["task_id"] == "agg-sum-all":
            built = candidate
            break
    if built is None:
        raise SystemExit("task not found")

    task_obj = _task_from(built)
    cfg = RunnerConfig(config_id="bench", seed=7, oracle="answer")

    print(f"task={task_obj.task_id} steps_needed~7 group={GROUP}")
    print(system_prompt(task_obj.tools)[:0] or "")
    rows = []
    rows.append(run_mode("A sequential", task_obj, cfg, concurrent=False, deadline_s=None))
    rows.append(run_mode("B concurrent", task_obj, cfg, concurrent=True, deadline_s=None))
    rows.append(run_mode("C conc+cut(20s)", task_obj, cfg, concurrent=True, deadline_s=20.0))

    out = Path("reports/rollout_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    seq = rows[0]["wall_s"]
    for row in rows[1:]:
        print(f"{row['mode']}: {seq/row['wall_s']:.2f}x faster than sequential, "
              f"solved {row['solved']}/{GROUP}")
    return 0


def _task_from(record: dict):
    """stage_io dumps tasks as JSON; rebuild the dataclass the runner expects."""

    from arlab.tasks.schema import Task

    return Task(
        task_id=record["task_id"],
        instruction=record["instruction"],
        required_fields=record.get("expected", {}),
        max_steps=record.get("max_steps", 8),
        tools=tuple(record.get("tools", ())),
        family=record.get("family", "lookup"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
