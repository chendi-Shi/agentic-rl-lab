"""Partial rollout executor: keep unfinished episodes across training rounds.

A training round has a fixed step budget. Two ways to handle an episode that is
longer than the budget:

* **truncate** -- cut it, score it as a failure, drop it. Simple, and wrong twice:
  the executed steps are wasted, and the reward is biased against long-horizon
  tasks, so the policy learns to avoid them.
* **partial rollout** -- keep the environment and trajectory alive, continue in a
  later round. Every step is executed exactly once and no episode is lost.

This module implements both over the same interface so the difference is a
measurement instead of an argument. It models *semantics* (what gets executed and
what survives); wall-clock effects belong to ``schedule_sim``.

The continuation is in-memory by design: an MCP server or a Docker container can
usually be kept alive for the duration of a run, but cannot be serialised, so
pretending to support cold resume would be a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from arlab.reward import compute_reward
from arlab.reward.answer import compute_task_reward
from arlab.rollout.runner import FinalAnswer, Policy, RunnerConfig
from arlab.store.trace import TraceRecord, TraceStore, rollout_key
from arlab.tasks.env import Environment
from arlab.tasks.schema import Step, Task, Trajectory


@dataclass
class EpisodeState:
    task: Task
    env: Environment
    traj: Trajectory
    finished: bool = False
    dropped: bool = False
    resumes: int = 0
    steps_executed: int = 0


@dataclass
class PartialStats:
    mode: str
    episodes: int = 0
    rounds: int = 0
    steps_executed: int = 0
    steps_wasted: int = 0
    finished: int = 0
    dropped: int = 0
    resumes: int = 0

    @property
    def waste_fraction(self) -> float:
        return self.steps_wasted / self.steps_executed if self.steps_executed else 0.0

    def summary(self) -> str:
        return (
            f"{self.mode:<14} rounds={self.rounds:3d}  finished={self.finished:4d} "
            f"dropped={self.dropped:3d}  steps={self.steps_executed:5d} "
            f"wasted={self.steps_wasted:4d} ({self.waste_fraction:5.1%})  resumes={self.resumes:4d}"
        )


def _execute(env: Environment, call) -> str:
    try:
        return env.call(call.name, call.args)
    except Exception as exc:  # noqa: BLE001 - tool failure is data, not a crash
        return f"ERROR: {type(exc).__name__}: {exc}"


class PartialRolloutDriver:
    """Advances every active episode by at most ``round_budget`` steps per round."""

    def __init__(
        self,
        tasks: list[Task],
        env_factory: Callable[[], Environment],
        policy: Policy,
        cfg: RunnerConfig = RunnerConfig(),
        *,
        policy_name: str = "policy",
        store: TraceStore | None = None,
        round_budget: int = 4,
        partial_rollout: bool = True,
        max_rounds: int = 64,
    ) -> None:
        if round_budget <= 0:
            raise ValueError("round_budget must be positive")
        self.tasks = list(tasks)
        self.env_factory = env_factory
        self.policy = policy
        self.cfg = cfg
        self.policy_name = policy_name
        self.store = store
        self.round_budget = round_budget
        self.partial_rollout = partial_rollout
        self.max_rounds = max_rounds

    # ---------------------------------------------------------------- internals

    def _advance(self, episode: EpisodeState, budget: int) -> None:
        """Execute at most ``budget`` steps, stopping early on completion."""

        for _ in range(budget):
            if episode.finished:
                return
            if len(episode.traj.steps) > episode.task.max_steps:
                episode.finished = True
                return

            action = self.policy(episode.task, episode.traj.steps)
            if action is None:
                episode.finished = True
                return
            if isinstance(action, FinalAnswer):
                episode.traj.final_answer = action.text
                episode.finished = True
                return

            observation = _execute(episode.env, action)
            episode.traj.steps.append(Step(tool_call=action, observation=observation))
            episode.steps_executed += 1

            if len(episode.traj.steps) > episode.task.max_steps:
                episode.finished = True
                return

    def _finalize(self, episode: EpisodeState) -> TraceRecord:
        if self.cfg.oracle == "state":
            breakdown = compute_reward(episode.task, episode.traj, episode.env.snapshot())
        else:
            breakdown = compute_task_reward(episode.task, episode.traj)

        key = rollout_key(episode.task.task_id, self.policy_name, self.cfg.config_id, self.cfg.seed)
        record = TraceRecord(
            key=key,
            task_id=episode.task.task_id,
            policy=self.policy_name,
            config_id=self.cfg.config_id,
            seed=self.cfg.seed,
            trajectory=episode.traj,
            reward=breakdown.as_dict(),
            meta={
                "policy": self.policy_name,
                "partial_rollout": self.partial_rollout,
                "resumes": episode.resumes,
                "round_budget": self.round_budget,
                "dropped": episode.dropped,
            },
        )
        if self.store is not None:
            self.store.append(record)
        return record

    # ------------------------------------------------------------------- public

    def run(self) -> PartialStats:
        stats = PartialStats(
            mode="partial" if self.partial_rollout else "truncate", episodes=len(self.tasks)
        )
        pending = list(self.tasks)
        active: list[EpisodeState] = []

        for round_index in range(self.max_rounds):
            while pending and len(active) < len(self.tasks):
                task = pending.pop(0)
                active.append(
                    EpisodeState(task=task, env=self.env_factory(), traj=Trajectory(task_id=task.task_id))
                )

            if not active and not pending:
                break
            stats.rounds = round_index + 1

            for episode in active:
                before = episode.steps_executed
                had_progress = before > 0
                self._advance(episode, self.round_budget)
                stats.steps_executed += episode.steps_executed - before
                # A "resume" is a carry-over that actually did more work. The
                # round whose only action is the policy saying "done" is not one.
                if had_progress and episode.steps_executed > before:
                    episode.resumes += 1
                    stats.resumes += 1

            still_active: list[EpisodeState] = []
            for episode in active:
                if episode.finished:
                    self._finalize(episode)
                    stats.finished += 1
                elif self.partial_rollout:
                    still_active.append(episode)
                else:
                    # Naive mode: score the truncated episode as a failure and drop it.
                    episode.dropped = True
                    stats.dropped += 1
                    stats.steps_wasted += episode.steps_executed
                    self._finalize(episode)
            active = still_active

            if not self.partial_rollout and not pending and not active:
                break

        # Anything still active after max_rounds is force-finished, not silently lost.
        for episode in active:
            episode.finished = True
            self._finalize(episode)
            stats.finished += 1

        return stats
