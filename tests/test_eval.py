from __future__ import annotations

import pytest

from arlab.eval import bootstrap_ci, paired_bootstrap


def test_detects_consistent_improvement():
    a = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    b = [0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]

    res = paired_bootstrap(a, b, n_boot=500, seed=1)

    assert res.mean_diff == pytest.approx(0.25)
    assert res.significant is True
    assert res.ci_low > 0


def test_identical_configs_are_not_significant():
    a = [0.0, 1.0, 0.5, 1.0, 0.0]
    res = paired_bootstrap(a, list(a), n_boot=500, seed=1)

    assert res.mean_diff == pytest.approx(0.0)
    assert res.significant is False


def test_mismatched_pairs_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0, 2.0], [1.0])


def test_empty_rejected():
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_bootstrap_ci_reports_spread():
    out = bootstrap_ci([0.2, 0.4, 0.6, 0.8], n_boot=500, seed=3)

    assert out["mean"] == pytest.approx(0.5)
    assert out["ci_low"] < out["mean"] < out["ci_high"]
