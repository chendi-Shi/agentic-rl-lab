"""Rollout loop: policy -> environment -> verifiable reward -> durable trace.

Framework-agnostic on purpose. Whether the policy is a scripted baseline, an
OpenAI-compatible vLLM endpoint, or a verl rollout worker, the loop, the budget
accounting and the reward computation stay identical -- which is what makes the
numbers comparable across SFT / RL / baseline.

Step-budget semantics: if the policy still wants to act once ``max_steps`` is
reached, that call is executed and recorded (so the trajectory is one step over
budget) and the loop stops. This is deliberate: it lets ``compute_reward`` see
``truncated=True`` instead of silently hiding the overrun.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from arlab.reward import compute_reward
from arlab.reward.answer import compute_answer_reward
from arlab.store.trace import TraceRecord, TraceStore, rollout_key
from arlab.tasks.schema import Step, Task, ToolCall, Trajectory
from arlab.tasks.env import Environment


@dataclass(frozen=True)
class FinalAnswer:
    """Returned by a policy to end the episode with a submitted answer."""

    text: str


Policy = Callable[[Task, list[Step]], "ToolCall | FinalAnswer | None"]


@dataclass(frozen=True)
class RunnerConfig:
    config_id: str = "default"
    seed: int = 0
    oracle: str = "state"  # "state" | "answer"

    def __post_init__(self) -> None:
        if self.oracle not in {"state", "answer"}:
            raise ValueError(f"unknown oracle: {self.oracle!r}")


@dataclass
class BatchResult:
    n_tasks: int = 0
    n_run: int = 0
    n_skipped: int = 0
    rewards: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0

    @property
    def mean_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    def summary(self) -> str:
        mean = f"{self.mean_reward:.4f}" if self.rewards else "n/a"
        return (
            f"tasks={self.n_tasks} run={self.n_run} skipped={self.n_skipped} "
            f"mean_reward={mean} wall={self.wall_seconds:.1f}s"
        )


def _execute(env: Environment, call: ToolCall) -> str:
    try:
        return env.call(call.name, call.args)
    except Exception as exc:  # noqa: BLE001 - a tool failure is data, not a crash
        return f"ERROR: {type(exc).__name__}: {exc}"


def run_task(
    task: Task,
    env_factory: Callable[[], Environment],
    policy: Policy,
    cfg: RunnerConfig = RunnerConfig(),
    *,
    policy_name: str = "policy",
    store: TraceStore | None = None,
) -> TraceRecord:
    """Run one episode. Returns the record; also persists it when ``store`` set."""

    key = rollout_key(task.task_id, policy_name, cfg.config_id, cfg.seed)
    if store is not None and store.is_done(key):
        raise AlreadyDone(f"{key} already in {store.path}")

    env = env_factory()
    traj = Trajectory(task_id=task.task_id)
    started = time.perf_counter()
    hit_step_limit = False

    while True:
        if len(traj.steps) > task.max_steps:
            hit_step_limit = True
            break
        action = policy(task, traj.steps)
        if action is None:
            break
        if isinstance(action, FinalAnswer):
            traj.final_answer = action.text
            break

        obs = _execute(env, action)
        traj.steps.append(Step(tool_call=action, observation=obs))

        if len(traj.steps) > task.max_steps:
            hit_step_limit = True
            break

    wall = time.perf_counter() - started

    if cfg.oracle == "state":
        breakdown = compute_reward(task, traj, env.snapshot())
    else:
        breakdown = compute_answer_reward(task, traj)

    meta = {
        "wall_seconds": round(wall, 3),
        "hit_step_limit": hit_step_limit,
        "policy": policy_name,
    }
    record = TraceRecord(
        key=key,
        task_id=task.task_id,
        policy=policy_name,
        config_id=cfg.config_id,
        seed=cfg.seed,
        trajectory=traj,
        reward=breakdown.as_dict(),
        meta=meta,
    )
    if store is not None:
        store.append(record)
    return record


class AlreadyDone(RuntimeError):
    """Raised when a rollout key is already present in the store."""


def run_batch(
    tasks: list[Task],
    env_factory: Callable[[], Environment],
    policy: Policy,
    cfg: RunnerConfig = RunnerConfig(),
    *,
    policy_name: str = "policy",
    store: TraceStore | None = None,
) -> BatchResult:
    """Run all tasks, skipping completed keys when a store is supplied."""

    result = BatchResult(n_tasks=len(tasks))
    started = time.perf_counter()
    for task in tasks:
        key = rollout_key(task.task_id, policy_name, cfg.config_id, cfg.seed)
        if store is not None and store.is_done(key):
            result.n_skipped += 1
            continue
        record = run_task(task, env_factory, policy, cfg, policy_name=policy_name, store=store)
        result.rewards.append(float(record.reward["total"]))
        result.n_run += 1
    result.wall_seconds = time.perf_counter() - started
    return result
