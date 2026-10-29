from __future__ import annotations

import pytest

from arlab.diagnostics import (
    classify_difficulty,
    detect_advantage_collapse,
    group_stats,
    length_percentiles,
)


def test_collapse_detected_when_all_groups_constant():
    rewards = {"t1": [1.0, 1.0, 1.0, 1.0], "t2": [0.0, 0.0, 0.0, 0.0]}
    report = detect_advantage_collapse(rewards)

    assert report.n_groups == 2
    assert report.n_collapsed == 2
    assert report.frac_collapsed == pytest.approx(1.0)
    assert report.healthy is False


def test_healthy_batch_passes():
    rewards = {
        "t1": [0.0, 0.5, 1.0, 0.5],
        "t2": [0.0, 1.0, 0.0, 1.0],
        "t3": [0.5, 0.5, 1.0, 0.0],
        "t4": [0.0, 0.0, 1.0, 0.0],
        "t5": [1.0, 0.5, 0.5, 0.0],
    }
    report = detect_advantage_collapse(rewards)

    assert report.n_collapsed == 0
    assert report.healthy is True


def test_partial_collapse_flagged_above_threshold():
    rewards = {
        "t1": [1.0, 1.0, 1.0, 1.0],
        "t2": [0.0, 1.0, 0.0, 1.0],
        "t3": [0.5, 0.5, 0.5, 0.5],
        "t4": [0.0, 1.0, 1.0, 0.0],
        "t5": [1.0, 0.0, 1.0, 0.0],
    }
    report = detect_advantage_collapse(rewards, max_frac=0.2)

    assert report.n_collapsed == 2
    assert report.frac_collapsed == pytest.approx(0.4)
    assert report.healthy is False
    assert report.collapsed_task_ids == ["t1", "t3"]


def test_single_sample_groups_are_ignored():
    report = detect_advantage_collapse({"t1": [1.0], "t2": [0.0, 1.0]})
    assert report.n_groups == 1


def test_difficulty_bands():
    rewards = {
        "impossible": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "learnable": [0.0, 1.0, 1.0, 0.0, 1.0],
        "trivial": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    report = classify_difficulty(rewards)

    assert "learnable" in report.kept
    assert "impossible" in report.too_hard
    assert "trivial" in report.too_easy


def test_group_stats_basic():
    stats = group_stats({"t1": [0.0, 1.0]})[0]
    assert stats.n == 2
    assert stats.mean == pytest.approx(0.5)
    assert stats.pass_rate == pytest.approx(0.5)


def test_length_tail_exposes_the_slow_batch():
    lengths = [3] * 95 + [40] * 5
    out = length_percentiles(lengths)

    assert out["p50"] == pytest.approx(3)
    assert out["p99"] >= 40
    # 5 long trajectories contribute a disproportionate share of total steps.
    assert out["tail_share"] > 0.3


def test_length_percentiles_empty():
    assert length_percentiles([])["mean"] == 0.0
