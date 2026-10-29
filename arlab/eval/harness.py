"""Paired comparison with bootstrap confidence intervals.

Agent evals are noisy and sample-starved: with 8B models on real tasks you often
get a handful of successes. Comparing two means with a t-test over different task
sets is a good way to publish a difference that is not there.

Pairing (same task, same seed, two configs) removes most task-difficulty variance,
and the bootstrap avoids distributional assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PairedResult:
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    prob_b_better: float
    significant: bool

    def summary(self) -> str:
        return (
            f"n={self.n}  A={self.mean_a:.4f}  B={self.mean_b:.4f}  "
            f"diff={self.mean_diff:+.4f}  95%CI=[{self.ci_low:+.4f}, {self.ci_high:+.4f}]  "
            f"P(B>A)={self.prob_b_better:.3f}  "
            f"{'SIGNIFICANT' if self.significant else 'not significant'}"
        )


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    """Bootstrap the mean of ``b - a`` over paired samples.

    ``significant`` is True only when the CI excludes zero -- report it, never
    round a non-significant difference into a claim.
    """

    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"paired samples must match: {arr_a.shape} vs {arr_b.shape}")
    if arr_a.size == 0:
        raise ValueError("empty samples")

    diffs = arr_b - arr_a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return PairedResult(
        n=int(diffs.size),
        mean_a=float(arr_a.mean()),
        mean_b=float(arr_b.mean()),
        mean_diff=float(diffs.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        prob_b_better=float((boot > 0).mean()),
        significant=bool(lo > 0 or hi < 0),
    )


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """CI for a single mean, e.g. pass rate over seeds."""

    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("empty sample")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n": float(arr.size),
        "mean": float(arr.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std": float(arr.std()),
    }
