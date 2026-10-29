"""Recording / replaying environment wrapper.

Two problems it solves, both of which cost GPU money if ignored:

1. **Determinism.** Tool output drift (a service returning different numbers)
   makes the reward noisy, and noise is what RL learns from badly.
2. **Offline development.** Once a run is recorded, the whole rollout loop can
   be replayed on CPU to debug reward logic without renting a card.

A miss policy decides what happens on replay when a call was never recorded:
``"error"`` (strict, default) or ``"live"`` (fall through to the wrapped env).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

from arlab.tasks.env import Environment

MissPolicy = Literal["error", "live"]


def _key(name: str, args: Mapping[str, Any]) -> str:
    return json.dumps({"name": name, "args": dict(args)}, sort_keys=True, ensure_ascii=False)


@dataclass
class RecordingEnvironment:
    """Calls the wrapped env, recording every (call -> observation) pair."""

    inner: Environment
    records: dict[str, str] = field(default_factory=dict)

    def tool_names(self) -> tuple[str, ...]:
        return self.inner.tool_names()

    def snapshot(self) -> dict[str, Any]:
        return self.inner.snapshot()

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        observation = self.inner.call(name, args)
        self.records[_key(name, args)] = observation
        return observation

    def dump(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, inner: Environment, path: str | Path) -> "RecordingEnvironment":
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(inner=inner, records=dict(records))


class ReplayEnvironment:
    """Serves recorded observations; optionally falls through to the live env."""

    def __init__(
        self,
        records: dict[str, str],
        inner: Environment | None = None,
        on_miss: MissPolicy = "error",
    ) -> None:
        if on_miss not in ("error", "live"):
            raise ValueError(f"unknown on_miss: {on_miss!r}")
        if on_miss == "live" and inner is None:
            raise ValueError("on_miss='live' requires an inner environment")
        self._records = records
        self._inner = inner
        self._on_miss = on_miss
        self.misses = 0

    def tool_names(self) -> tuple[str, ...]:
        if self._inner is not None:
            return self._inner.tool_names()
        names = {json.loads(k)["name"] for k in self._records}
        return tuple(sorted(names))

    def snapshot(self) -> dict[str, Any]:
        if self._inner is None:
            return {}
        return self._inner.snapshot()

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        key = _key(name, args)
        if key in self._records:
            return self._records[key]
        self.misses += 1
        if self._on_miss == "live" and self._inner is not None:
            return self._inner.call(name, args)
        raise KeyError(f"no recorded observation for {name}({dict(args)})")

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        inner: Environment | None = None,
        on_miss: MissPolicy = "error",
    ) -> "ReplayEnvironment":
        records = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(records, inner=inner, on_miss=on_miss)
