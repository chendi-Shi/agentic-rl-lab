"""Partial rollout, tested without a GPU.

The point of these tests is the contract, not the numbers: a policy that returns
``None`` must end the episode cleanly, keep the steps it already produced, still be
scored, and not slow the rest of the group down.
"""

from __future__ import annotations

import time

import pytest

from arlab.rollout.deadline import DeadlinePolicy, collect_with_deadline
from arlab.rollout.runner import RunnerConfig
from arlab.tasks.env import SyntheticBusinessEnv
from arlab.tasks.schema import Task, ToolCall


def make_task(max_steps: int = 6) -> Task:
    return Task(
        task_id="t1",
        instruction="look up A123 and answer",
        required_fields={"answer": "199"},
        max_steps=max_steps,
        tools=("orders.get",),
        family="business",
    )


def slow_then_finish(delay: float, steps_to_run: int):
    """A policy that sleeps on every call, so the deadline has something to cut."""

    def policy(task, steps):
        time.sleep(delay)
        if len(steps) >= steps_to_run:
            return __import__("arlab.rollout.runner", fromlist=["FinalAnswer"]).FinalAnswer(
                "199"
            )
        return ToolCall("orders.get", {"order_id": "A123"})

    return policy


def test_deadline_policy_returns_none_after_expiry():
    inner = slow_then_finish(0.0, 99)
    policy = DeadlinePolicy(inner, deadline_s=0.05)
    time.sleep(0.06)

    assert policy(None, []) is None
    assert policy.truncated is True


def test_deadline_policy_passes_through_before_expiry():
    inner = slow_then_finish(0.0, 99)
    policy = DeadlinePolicy(inner, deadline_s=10.0)

    action = policy(None, [])
    assert isinstance(action, ToolCall)
    assert policy.truncated is False


def test_truncated_episode_is_still_scored_and_keeps_its_steps():
    """A cut episode is a normal, scorable record -- that is the contract."""

    task = make_task()
    cfg = RunnerConfig(config_id="cut", seed=0, oracle="state")

    def policy_factory(seed):
        return slow_then_finish(0.02, steps_to_run=99)  # never finishes

    records, stats = collect_with_deadline(
        task,
        SyntheticBusinessEnv,
        policy_factory,
        cfg,
        group_size=2,
        workers=2,
        deadline_s=0.05,
    )

    assert len(records) == 2
    for record in records:
        assert record.trajectory.steps, "the steps taken before the deadline must be kept"
        assert "total" in record.reward
    assert stats.truncated == 2


def test_deadline_bounds_wall_time():
    """Without a deadline this policy runs to max_steps; with one it stops early."""

    task = make_task(max_steps=30)
    cfg = RunnerConfig(config_id="bound", seed=0, oracle="state")

    def policy_factory(seed):
        return slow_then_finish(0.01, steps_to_run=10_000)

    started = time.perf_counter()
    _, uncut = collect_with_deadline(
        task, SyntheticBusinessEnv, policy_factory, cfg,
        group_size=4, workers=4, deadline_s=None,
    )
    uncut_wall = time.perf_counter() - started

    started = time.perf_counter()
    _, cut = collect_with_deadline(
        task, SyntheticBusinessEnv, policy_factory, cfg,
        group_size=4, workers=4, deadline_s=0.08,
    )
    cut_wall = time.perf_counter() - started

    assert cut.truncated == 4
    assert cut_wall < uncut_wall
    assert uncut.solved == cut.solved == 0


def test_fast_group_is_untouched_by_a_generous_deadline():
    task = make_task()
    cfg = RunnerConfig(config_id="fast", seed=0, oracle="state")

    def policy_factory(seed):
        return slow_then_finish(0.0, steps_to_run=1)  # answers immediately

    records, stats = collect_with_deadline(
        task, SyntheticBusinessEnv, policy_factory, cfg,
        group_size=4, workers=4, deadline_s=5.0,
    )

    assert stats.truncated == 0
    assert all(record.reward["total"] >= 0 for record in records)


def test_stats_summary_reports_percentiles():
    task = make_task()
    cfg = RunnerConfig(config_id="pct", seed=0, oracle="state")

    def policy_factory(seed):
        return slow_then_finish(0.01, steps_to_run=1)

    _, stats = collect_with_deadline(
        task, SyntheticBusinessEnv, policy_factory, cfg,
        group_size=4, workers=4, deadline_s=None,
    )

    assert stats.n == 4
    assert stats.percentile(0.99) >= stats.percentile(0.5)
    assert "p99" in stats.summary()
