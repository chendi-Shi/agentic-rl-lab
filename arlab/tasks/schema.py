"""Trajectory / task schema shared by every module (framework-agnostic)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation emitted by the policy."""

    name: str
    args: Mapping[str, Any]

    def canonical(self) -> tuple[str, str]:
        """Stable identity, used for duplicate-call detection."""
        return (self.name, json.dumps(dict(self.args), sort_keys=True, ensure_ascii=False))


@dataclass
class Step:
    """One turn of the agent loop.

    ``tool_call`` may be ``None`` for a pure reasoning step; such steps consume
    budget but earn no format credit. Token counts are recorded per step so the
    cost of a trajectory -- not just its score -- is auditable.
    """

    tool_call: ToolCall | None = None
    observation: str | None = None
    note: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class Trajectory:
    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    def tokens(self) -> tuple[int, int]:
        return (
            sum(s.tokens_in for s in self.steps),
            sum(s.tokens_out for s in self.steps),
        )

    def __len__(self) -> int:
        return len(self.steps)


@dataclass(frozen=True)
class Task:
    """A verifiable task.

    ``required_fields`` maps environment state keys to their expected values.
    The oracle never reads the model's prose: it only inspects environment
    state, which is what makes the reward unspoofable.
    """

    task_id: str
    instruction: str
    required_fields: Mapping[str, Any]
    max_steps: int = 12
    tools: tuple[str, ...] = ()
    family: str = "default"

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not self.required_fields:
            raise ValueError("required_fields must not be empty")
