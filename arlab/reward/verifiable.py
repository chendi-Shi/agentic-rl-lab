"""Four-part verifiable reward for multi-turn tool-use agents.

This is the piece that decides whether the whole RL run learns anything.
Design constraints, each of which exists because it kills a real failure mode:

1. ``r_task`` reads **environment state only** -- never the model's prose.
   A model that claims "I refunded it" without calling the tool scores 0.
2. ``r_task`` gives **partial credit** over required fields, so a group of
   samples does not collapse to all-zero and destroy the GRPO advantage.
3. ``r_format`` dominates early and decays, so syntax is learned first and does
   not crowd out task reward later.
4. Duplicate and invalid calls are penalised, so "call the same tool forever"
   is not a viable policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from arlab.tasks.schema import Task, ToolCall, Trajectory


@dataclass(frozen=True)
class RewardConfig:
    field_weight: float = 1.0
    format_weight: float = 0.1
    format_warmup_steps: int = 3
    format_late_scale: float = 0.3
    invalid_penalty: float = 0.2
    duplicate_penalty: float = 0.1
    step_penalty: float = 0.01
    tool_error_penalty: float = 0.05
    truncation_penalty: float = 0.5
    error_prefix: str = "ERROR:"


DEFAULT_CONFIG = RewardConfig()


@dataclass
class RewardBreakdown:
    """Total plus every component, so a flat curve is diagnosable."""

    total: float
    r_task: float
    r_format: float
    p_invalid: float
    p_duplicate: float
    p_steps: float
    p_error: float
    p_truncation: float
    matched_fields: int
    total_fields: int
    n_calls: int
    n_steps: int
    truncated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "r_task": self.r_task,
            "r_format": self.r_format,
            "p_invalid": self.p_invalid,
            "p_duplicate": self.p_duplicate,
            "p_steps": self.p_steps,
            "p_error": self.p_error,
            "p_truncation": self.p_truncation,
            "matched_fields": self.matched_fields,
            "total_fields": self.total_fields,
            "n_calls": self.n_calls,
            "n_steps": self.n_steps,
            "truncated": self.truncated,
        }


def _values_equal(actual: Any, expected: Any, tol: float = 1e-6) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return bool(actual) is bool(expected)
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= tol
    return actual == expected


def _is_valid_call(call: ToolCall, tools: tuple[str, ...]) -> bool:
    if call.name not in tools:
        return False
    return len(dict(call.args)) > 0


def compute_reward(
    task: Task,
    traj: Trajectory,
    state: Mapping[str, Any],
    cfg: RewardConfig = DEFAULT_CONFIG,
) -> RewardBreakdown:
    """Score one trajectory against the environment's final state."""

    required = dict(task.required_fields)
    matched = sum(
        1 for key, expected in required.items() if _values_equal(state.get(key), expected)
    )
    total_fields = len(required)

    n_steps = len(traj)
    truncated = n_steps > task.max_steps

    # Truncation counts as failure: the policy must learn to finish in budget.
    r_task = 0.0 if truncated else matched / total_fields

    calls = traj.tool_calls()
    n_calls = len(calls)

    # Format reward: valid-call ratio, weighted to matter early and decay late.
    weighted_valid = 0.0
    weight_sum = 0.0
    for index, step in enumerate(traj.steps):
        if step.tool_call is None:
            continue
        weight = 1.0 if index < cfg.format_warmup_steps else cfg.format_late_scale
        weight_sum += weight
        if _is_valid_call(step.tool_call, task.tools):
            weighted_valid += weight
    r_format = (weighted_valid / weight_sum) if weight_sum > 0 else 0.0

    invalid = sum(1 for call in calls if not _is_valid_call(call, task.tools))

    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for call in calls:
        key = call.canonical()
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    errors = sum(
        1 for step in traj.steps if (step.observation or "").startswith(cfg.error_prefix)
    )

    p_invalid = cfg.invalid_penalty * invalid
    p_duplicate = cfg.duplicate_penalty * duplicates
    p_steps = cfg.step_penalty * n_steps
    p_error = cfg.tool_error_penalty * errors
    p_truncation = cfg.truncation_penalty if truncated else 0.0

    total = (
        cfg.field_weight * r_task
        + cfg.format_weight * r_format
        - p_invalid
        - p_duplicate
        - p_steps
        - p_error
        - p_truncation
    )

    return RewardBreakdown(
        total=total,
        r_task=r_task,
        r_format=r_format,
        p_invalid=p_invalid,
        p_duplicate=p_duplicate,
        p_steps=p_steps,
        p_error=p_error,
        p_truncation=p_truncation,
        matched_fields=matched,
        total_fields=total_fields,
        n_calls=n_calls,
        n_steps=n_steps,
        truncated=truncated,
    )
