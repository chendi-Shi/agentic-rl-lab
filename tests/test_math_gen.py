from __future__ import annotations

import pytest

from arlab.reward.answer import answers_match
from arlab.tasks.math_gen import TIERS, generate, tier_counts, verify


def test_generates_balanced_tiers():
    tasks = generate(per_tier=5, seed=7)
    counts = tier_counts(tasks)

    assert len(tasks) == 15
    assert counts == {tier: 5 for tier in TIERS}


def test_every_expression_matches_its_answer():
    """The generator's own self-check: this is what protects the reward signal."""
    tasks = generate(per_tier=8, seed=3)
    assert verify(tasks) == []


def test_ids_and_instructions_are_unique():
    tasks = generate(per_tier=6, seed=1)

    assert len({t.task_id for t in tasks}) == len(tasks)
    assert len({t.instruction for t in tasks}) == len(tasks)


def test_deterministic_for_a_seed():
    assert generate(per_tier=4, seed=11) == generate(per_tier=4, seed=11)
    assert generate(per_tier=4, seed=11) != generate(per_tier=4, seed=12)


def test_answers_survive_the_numeric_matcher():
    """The answer the task carries must be accepted by the scorer itself."""
    for task in generate(per_tier=3, seed=5):
        assert answers_match(task.answer, task.answer), task.task_id
        assert answers_match(f"结果是 {task.answer}", task.answer), task.task_id


def test_instructions_carry_the_tool_hint():
    for task in generate(per_tier=2, seed=2):
        assert "python" in task.instruction


def test_bad_args_rejected():
    with pytest.raises(ValueError):
        generate(per_tier=0)


def test_difficulty_is_ordered_by_tier():
    """Sanity: hard tasks should not be tiny numbers like the easy ones."""
    tasks = generate(per_tier=6, seed=9)
    easy = [float(t.answer) for t in tasks if t.tier == "easy"]
    hard = [float(t.answer) for t in tasks if t.tier == "hard"]
    assert max(len(str(int(v))) for v in easy) <= 5
    assert any(v > 200 for v in hard)
