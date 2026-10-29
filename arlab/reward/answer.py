"""Reward for answer-submitting tasks (stage 1: QA / math-with-code-interpreter).

Same penalty structure as the state oracle, so numbers from stage 1 and stage 2
are comparable; only the task signal differs (normalised exact match instead of
environment state).
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from arlab.reward.verifiable import DEFAULT_CONFIG, RewardBreakdown, RewardConfig
from arlab.tasks.schema import Task, Trajectory

_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def normalize(text: str) -> str:
    """Lowercase, unify width, strip punctuation noise and whitespace."""

    text = unicodedata.normalize("NFKC", text or "").lower().strip()
    text = text.replace(",", "")
    text = _WS.sub(" ", text)
    return text.strip(" .。!！?？\"'`")


def extract_number(text: str) -> str | None:
    """Pull the last number out of a free-form answer.

    Agents wrap answers in prose ("the result is 400"), so exact string match
    alone would punish formatting rather than reasoning.
    """

    matches = _NUM.findall(text or "")
    return matches[-1] if matches else None


def answers_match(predicted: str | None, expected: str, numeric: bool = True) -> bool:
    if predicted is None:
        return False
    if normalize(predicted) == normalize(expected):
        return True
    # "400" vs "400.0": only compare as numbers when both sides are numeric.
    p_num, e_num = extract_number(predicted or ""), extract_number(expected or "")
    if numeric and p_num is not None and e_num is not None:
        try:
            return abs(float(p_num) - float(e_num)) <= 1e-6
        except ValueError:
            return False
    return False


@dataclass(frozen=True)
class AnswerTaskSpec:
    """Adapter so a plain ``Task`` plus ``expected_answer`` can be scored."""

    expected_answer: str
    numeric: bool = True


def compute_answer_reward(
    task: Task,
    traj: Trajectory,
    expected_answer: str | None = None,
    cfg: RewardConfig = DEFAULT_CONFIG,
    numeric: bool = True,
) -> RewardBreakdown:
    """Score a trajectory by its submitted final answer."""

    expected = expected_answer or str(task.required_fields.get("answer", ""))
    if not expected:
        raise ValueError(f"task {task.task_id} has no expected answer")

    truncated = len(traj) > task.max_steps
    matched = (not truncated) and answers_match(traj.final_answer, expected, numeric=numeric)

    return _assemble(task, traj, 1.0 if matched else 0.0, int(matched), 1, truncated, cfg)


def _field_score(actual: Any, expected: Any) -> float:
    """Per-field credit in [0, 1]; lists score by set overlap."""

    if expected is None:
        return 1.0 if actual is None else 0.0
    if actual is None:
        return 0.0
    if isinstance(expected, bool) or isinstance(actual, bool):
        return 1.0 if bool(actual) is bool(expected) else 0.0
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            tolerance = max(1e-6, abs(float(expected)) * 0.01)
            return 1.0 if abs(float(actual) - float(expected)) <= tolerance else 0.0
        return 0.0
    if isinstance(expected, list):
        if not isinstance(actual, list) or not expected:
            return 0.0
        got = {normalize(str(x)) for x in actual}
        want = {normalize(str(x)) for x in expected}
        return len(got & want) / len(want)
    return 1.0 if normalize(str(actual)) == normalize(str(expected)) else 0.0


def compute_structured_answer_reward(
    task: Task,
    traj: Trajectory,
    expected: dict[str, Any],
    cfg: RewardConfig = DEFAULT_CONFIG,
    weights: dict[str, float] | None = None,
) -> RewardBreakdown:
    """Dense reward for a structured JSON answer (stage 2: incident diagnosis).

    The platform's answer schema has seven fields (``service``, ``error_rate``,
    ``status``, ``likely_cause``, ``recommendation``, ``citations``,
    ``uncertainty``). Scoring them individually is what keeps a group from
    collapsing: a partial diagnosis earns partial credit.
    """

    truncated = len(traj) > task.max_steps
    parsed: Any = None
    if traj.final_answer:
        try:
            parsed = json.loads(traj.final_answer)
        except (json.JSONDecodeError, TypeError):
            parsed = None

    if not isinstance(parsed, dict):
        # Unparseable answer: no task credit, penalties still apply.
        return _assemble(task, traj, 0.0, 0, len(expected), truncated, cfg)

    field_weights = dict(weights or {})
    total_weight = 0.0
    earned = 0.0
    exact = 0
    for field, want in expected.items():
        weight = float(field_weights.get(field, 1.0))
        total_weight += weight
        score = _field_score(parsed.get(field), want)
        earned += weight * score
        if score >= 0.999:
            exact += 1

    r_task = 0.0 if truncated else (earned / total_weight if total_weight else 0.0)
    return _assemble(
        task, traj, r_task, 0 if truncated else exact, len(expected), truncated, cfg
    )


def compute_task_reward(
    task: Task, traj: Trajectory, cfg: RewardConfig = DEFAULT_CONFIG
) -> RewardBreakdown:
    """Dispatch on the task shape: a single ``answer`` field or a JSON object.

    Stage-1 math tasks submit one value; stage-2 diagnosis tasks submit the
    seven-field object. The runner should not have to know which is which.
    """

    if set(task.required_fields) == {"answer"}:
        return compute_answer_reward(task, traj, cfg=cfg)
    return compute_structured_answer_reward(task, traj, dict(task.required_fields), cfg=cfg)


def _assemble(
    task: Task,
    traj: Trajectory,
    r_task: float,
    matched: int,
    total: int,
    truncated: bool,
    cfg: RewardConfig,
) -> RewardBreakdown:
    """Shared penalty accounting, kept in one place so both oracles agree."""

    total_steps = len(traj)
    calls = traj.tool_calls()

    weighted_valid = 0.0
    weight_sum = 0.0
    for index, step in enumerate(traj.steps):
        if step.tool_call is None:
            continue
        weight = 1.0 if index < cfg.format_warmup_steps else cfg.format_late_scale
        weight_sum += weight
        valid = step.tool_call.name in task.tools and len(dict(step.tool_call.args)) > 0
        if valid:
            weighted_valid += weight
    r_format = (weighted_valid / weight_sum) if weight_sum > 0 else 0.0

    invalid = sum(1 for c in calls if c.name not in task.tools or not dict(c.args))
    seen: set[Any] = set()
    duplicates = 0
    for call in calls:
        key = call.canonical()
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)
    errors = sum(1 for s in traj.steps if (s.observation or "").startswith(cfg.error_prefix))

    p_invalid = cfg.invalid_penalty * invalid
    p_duplicate = cfg.duplicate_penalty * duplicates
    p_steps = cfg.step_penalty * total_steps
    p_error = cfg.tool_error_penalty * errors
    p_truncation = cfg.truncation_penalty if truncated else 0.0

    total_reward = (
        cfg.field_weight * r_task
        + cfg.format_weight * r_format
        - p_invalid
        - p_duplicate
        - p_steps
        - p_error
        - p_truncation
    )

    return RewardBreakdown(
        total=total_reward,
        r_task=r_task,
        r_format=r_format,
        p_invalid=p_invalid,
        p_duplicate=p_duplicate,
        p_steps=p_steps,
        p_error=p_error,
        p_truncation=p_truncation,
        matched_fields=matched,
        total_fields=total,
        n_calls=len(calls),
        n_steps=total_steps,
        truncated=truncated,
    )
