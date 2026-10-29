"""Append-only JSONL trace store with resume support.

Renting a GPU by the hour makes a re-runnable experiment mandatory: a crashed
sweep must continue where it stopped instead of re-billing completed work, and
every number in the README must be reconstructable from disk.

Records are keyed by ``(task_id, policy, config_id, seed)`` so resume is exact.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from arlab.tasks.schema import Step, ToolCall, Trajectory


def rollout_key(task_id: str, policy: str, config_id: str, seed: int) -> str:
    return f"{task_id}|{policy}|{config_id}|{seed}"


@dataclass
class TraceRecord:
    key: str
    task_id: str
    policy: str
    config_id: str
    seed: int
    trajectory: Trajectory
    reward: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        tokens_in, tokens_out = self.trajectory.tokens()
        return {
            "key": self.key,
            "task_id": self.task_id,
            "policy": self.policy,
            "config_id": self.config_id,
            "seed": self.seed,
            "reward": self.reward,
            "final_answer": self.trajectory.final_answer,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "meta": self.meta,
            "steps": [
                {
                    "tool": step.tool_call.name if step.tool_call else None,
                    "args": dict(step.tool_call.args) if step.tool_call else None,
                    "observation": step.observation,
                    "note": step.note,
                    "tokens_in": step.tokens_in,
                    "tokens_out": step.tokens_out,
                }
                for step in self.trajectory.steps
            ],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TraceRecord":
        steps = [
            Step(
                tool_call=ToolCall(s["tool"], s.get("args") or {}) if s.get("tool") else None,
                observation=s.get("observation"),
                note=s.get("note"),
                tokens_in=int(s.get("tokens_in", 0)),
                tokens_out=int(s.get("tokens_out", 0)),
            )
            for s in payload.get("steps", [])
        ]
        traj = Trajectory(
            task_id=payload["task_id"],
            steps=steps,
            final_answer=payload.get("final_answer"),
        )
        return cls(
            key=payload["key"],
            task_id=payload["task_id"],
            policy=payload["policy"],
            config_id=payload["config_id"],
            seed=int(payload["seed"]),
            trajectory=traj,
            reward=payload.get("reward") or {},
            meta=payload.get("meta") or {},
        )


class TraceStore:
    """Durable, order-independent log of rollouts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[str] | None = None

    def done_keys(self) -> set[str]:
        """Keys already on disk; cached, since a sweep only grows the file."""

        if self._done is None:
            self._done = {r.key for r in self.load()}
        return self._done

    def is_done(self, key: str) -> bool:
        return key in self.done_keys()

    def append(self, record: TraceRecord) -> None:
        line = json.dumps(record.to_json(), ensure_ascii=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.done_keys().add(record.key)

    def load(self) -> list[TraceRecord]:
        if not self.path.exists():
            return []
        out: list[TraceRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(TraceRecord.from_json(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as exc:
                    # A truncated final line is expected after a hard kill.
                    raise ValueError(f"corrupt trace at {self.path}:{lineno}: {exc}") from exc
        return out

    def rewards_by_task(self) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = {}
        for record in self.load():
            grouped.setdefault(record.task_id, []).append(float(record.reward.get("total", 0.0)))
        return grouped


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
