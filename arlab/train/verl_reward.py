"""Reward entry point in the shape RL trainers call.

verl (and most GRPO implementations) call a single scoring function per sample:

    compute_score(data_source, solution_str, ground_truth, extra_info) -> {"score": float}

That signature is stable across verl releases, unlike the multi-turn agent API,
so this file is the part that survives an upgrade. It reuses the same comparison
logic as the offline reward, which means an offline baseline and the trainer
cannot disagree about what "correct" means.

``ground_truth`` is the JSON string written by ``arlab/train/export.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from arlab.reward.answer import answers_match, field_score

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str | None) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply, or None."""

    if not text:
        return None
    candidate = text
    match = _FENCE.search(text)
    if match:
        candidate = match.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def score_answer(solution_str: str, expected: dict[str, Any]) -> float:
    """Single-value answer: 1.0 for a match, else 0.0."""

    target = str(expected.get("answer", ""))
    return 1.0 if answers_match(solution_str, target) else 0.0


def score_structured(solution_str: str, expected: dict[str, Any]) -> float:
    """Object answer: mean per-field credit, missing/invalid JSON scores 0."""

    parsed = extract_json(solution_str)
    if parsed is None:
        return 0.0
    if not expected:
        return 0.0
    return sum(field_score(parsed.get(key), want) for key, want in expected.items()) / len(expected)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, float]:
    """Trainer entry point. Never raises: a bad reply is a zero, not a crash."""

    try:
        expected = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"score": 0.0, "acc": 0.0, "malformed_ground_truth": 1.0}

    if not isinstance(expected, dict) or not expected:
        return {"score": 0.0, "acc": 0.0, "malformed_ground_truth": 1.0}

    if set(expected) == {"answer"}:
        score = score_answer(solution_str, expected)
    else:
        score = score_structured(solution_str, expected)

    return {"score": float(score), "acc": 1.0 if score >= 0.999 else 0.0}
