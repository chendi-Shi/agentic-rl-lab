"""CPU end-to-end smoke test for both stages, with resume.

    python scripts/smoke_rollout.py

Runs twice on purpose. The second pass must skip everything already on disk,
which is the property that keeps a crashed GPU sweep from re-billing work.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.diagnostics import (  # noqa: E402
    classify_difficulty,
    detect_advantage_collapse,
    length_percentiles,
)
from arlab.rollout import FinalAnswer, RunnerConfig, run_batch  # noqa: E402
from arlab.store import TraceStore  # noqa: E402
from arlab.tasks.business_tasks import ORDER_OF, make_business_tasks  # noqa: E402
from arlab.tasks.env import SyntheticBusinessEnv  # noqa: E402
from arlab.tasks.math_env import MathEnv, make_math_tasks  # noqa: E402
from arlab.tasks.schema import Task, ToolCall  # noqa: E402

STORE = Path("traces/smoke.jsonl")

# --------------------------------------------------------------------------- #
# stage 1: math with a code interpreter (answer oracle)
# --------------------------------------------------------------------------- #

MATH_CODE = {
    "m-01": "17*23+9",
    "m-02": "(144/12)**2-19",
    "m-03": "sum(range(1,51))",
    "m-04": "2**10-24",
    "m-05": "(12-2)*180",
    "m-06": "6*5*4*3*2*1+5*4*3*2*1",
}


def math_policy(task: Task, steps):
    if not steps:
        return ToolCall("python", {"code": MATH_CODE[task.task_id]})
    return FinalAnswer(steps[0].observation.strip())


def guess_policy(task: Task, steps):
    return FinalAnswer("42")


# --------------------------------------------------------------------------- #
# stage 2: business actions (state oracle)
# --------------------------------------------------------------------------- #


def business_policy(task: Task, steps):
    order = ORDER_OF[task.task_id]
    if not steps:
        return ToolCall("orders.get", {"order_id": order})
    if task.task_id.startswith("reject"):
        return None
    if len(steps) == 1:
        return ToolCall("orders.refund", {"order_id": order})
    return None


def claim_only_policy(task: Task, steps):
    return FinalAnswer("已完成")


def main() -> int:
    store = TraceStore(STORE)
    stage1 = make_math_tasks()
    stage2 = make_business_tasks()

    print("=" * 74)
    print("stage 1 - math + code interpreter (answer oracle)")
    print("=" * 74)
    cfg1 = RunnerConfig(config_id="stage1-smoke", seed=0, oracle="answer")
    for name, policy in (("math_correct", math_policy), ("guess", guess_policy)):
        result = run_batch(
            stage1, MathEnv, policy, cfg1, policy_name=name, store=store
        )
        print(f"  {name:<16}{result.summary()}")

    print()
    print("=" * 74)
    print("stage 2 - business actions (state oracle)")
    print("=" * 74)
    cfg2 = RunnerConfig(config_id="stage2-smoke", seed=0, oracle="state")
    for name, policy in (("business_ok", business_policy), ("claim_only", claim_only_policy)):
        result = run_batch(
            stage2, SyntheticBusinessEnv, policy, cfg2, policy_name=name, store=store
        )
        print(f"  {name:<16}{result.summary()}")

    print()
    print("=" * 74)
    print("resume check - identical sweep, nothing should be re-billed")
    print("=" * 74)
    again = run_batch(stage1, MathEnv, math_policy, cfg1, policy_name="math_correct", store=store)
    print(f"  {again.summary()}")

    print()
    print("=" * 74)
    print("diagnostics over everything on disk")
    print("=" * 74)
    rewards = store.rewards_by_task()
    collapse = detect_advantage_collapse(rewards)
    print(f"  {collapse.summary()}")
    diff = classify_difficulty(rewards)
    print(f"  keep    : {diff.kept or '-'}")
    print(f"  too easy: {diff.too_easy or '-'}")
    print(f"  too hard: {diff.too_hard or '-'}")

    lengths = [len(r.trajectory) for r in store.load()]
    tail = length_percentiles(lengths)
    print(f"  length  : p50={tail['p50']:.0f} p90={tail['p90']:.0f} p99={tail['p99']:.0f}")

    records = store.load()
    tokens_in = sum(r.trajectory.tokens()[0] for r in records)
    tokens_out = sum(r.trajectory.tokens()[1] for r in records)
    print(f"  traces  : {len(records)} records, tokens_in={tokens_in} tokens_out={tokens_out}")
    print(f"  store   : {STORE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
