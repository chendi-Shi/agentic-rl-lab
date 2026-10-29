from __future__ import annotations

import pytest

from arlab.rollout.schedule_sim import (
    compare,
    pareto_lengths,
    simulate_async_refill,
    simulate_sync_batch,
    simulate_truncation,
    tail_share,
)


def test_async_never_slower_than_sync_on_long_tail():
    lengths = pareto_lengths(64, seed=1)
    sync = simulate_sync_batch(lengths, concurrency=8)
    async_ = simulate_async_refill(lengths, concurrency=8)

    assert async_.wall_ms <= sync.wall_ms
    assert async_.slot_utilization >= sync.slot_utilization


def test_identical_work_in_both_modes():
    """Scheduling must not change the amount of compute, only when it happens."""
    lengths = pareto_lengths(32, seed=2)
    sync = simulate_sync_batch(lengths, 8, step_time_ms=10.0)
    async_ = simulate_async_refill(lengths, 8, step_time_ms=10.0)

    assert sync.total_step_ms == pytest.approx(async_.total_step_ms)
    assert sync.n_episodes == async_.n_episodes == len(lengths)


def test_uniform_lengths_make_both_modes_equivalent():
    lengths = [5] * 16
    sync = simulate_sync_batch(lengths, 8)
    async_ = simulate_async_refill(lengths, 8)

    assert sync.wall_ms == pytest.approx(async_.wall_ms)
    assert sync.slot_utilization == pytest.approx(async_.slot_utilization)


def test_sync_idles_when_one_episode_is_long():
    """Eight episodes on eight slots: one long episode stalls the whole batch."""
    lengths = [30, 2, 2, 2, 2, 2, 2, 2]
    sync = simulate_sync_batch(lengths, 8)

    # 44 steps of real work spread over 8 slots; the batch advances at 30 steps.
    assert sync.wall_ms == pytest.approx(30)
    assert sync.slot_utilization == pytest.approx(44 / (30 * 8))
    # Every short episode waits 28 steps behind the long one.
    assert sync.idle_slot_ms == pytest.approx(7 * 28)


def test_refill_helps_only_when_work_is_queued():
    """With as many slots as episodes there is nothing to refill."""
    lengths = [30, 2, 2, 2, 2, 2, 2, 2]
    assert simulate_async_refill(lengths, 8).slot_utilization == pytest.approx(
        simulate_sync_batch(lengths, 8).slot_utilization
    )

    queued = lengths + [2] * 8
    assert simulate_async_refill(queued, 8).slot_utilization > simulate_sync_batch(
        queued, 8
    ).slot_utilization


def test_utilization_is_bounded():
    lengths = pareto_lengths(40, seed=3)
    for result in compare(lengths, 8).values():
        assert 0.0 < result.slot_utilization <= 1.0
        assert result.idle_slot_ms >= 0.0


def test_truncation_loses_long_episodes():
    lengths = [3, 5, 9, 12]
    cut = simulate_truncation(lengths, round_budget=8, partial_rollout=False)

    assert cut.finished == 2
    assert cut.truncated == 2
    assert cut.executed_steps == 3 + 5 + 8 + 8
    assert cut.wasted_steps == 16
    assert cut.waste_fraction == pytest.approx(16 / 24)


def test_partial_rollout_executes_every_step_once():
    lengths = [3, 5, 9, 12]
    partial = simulate_truncation(lengths, round_budget=8, partial_rollout=True)

    assert partial.finished == len(lengths)
    assert partial.truncated == 0
    assert partial.executed_steps == sum(lengths)
    assert partial.wasted_steps == 0


def test_truncation_validation():
    with pytest.raises(ValueError):
        simulate_truncation([], 8, partial_rollout=True)
    with pytest.raises(ValueError):
        simulate_sync_batch([1], concurrency=0)


def test_tail_share_is_concentrated():
    lengths = [2] * 95 + [40] * 5
    assert tail_share(lengths) > 0.3
    assert tail_share([5] * 10) == pytest.approx(0.1)


def test_pareto_lengths_are_bounded_and_seeded():
    a = pareto_lengths(50, seed=7)
    b = pareto_lengths(50, seed=7)
    assert a == b
    assert all(1 <= value <= 40 for value in a)
