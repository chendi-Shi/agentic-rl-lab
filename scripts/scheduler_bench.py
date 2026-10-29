"""How much throughput is lost to long-tailed agent episodes.

    python scripts/scheduler_bench.py
    python scripts/scheduler_bench.py --episodes 128 --concurrency 8 --step-ms 320

No GPU required: this is a closed-loop model over episode lengths, so it puts a
number on the infra problem *before* you rent a card. Replace ``--step-ms`` with
a measured value on the real run and the ratio stays meaningful.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.rollout.schedule_sim import (  # noqa: E402
    compare,
    pareto_lengths,
    simulate_truncation,
    tail_share,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--step-ms", type=float, default=1.0)
    parser.add_argument("--round-budget", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    lengths = pareto_lengths(args.episodes, seed=args.seed)

    print("=" * 72)
    print(f"episodes={len(lengths)}  concurrency={args.concurrency}  "
          f"total_steps={sum(lengths)}  tail_share(5%)={tail_share(lengths):.1%}")
    print(f"length: min={min(lengths)} median={sorted(lengths)[len(lengths) // 2]} max={max(lengths)}")
    print("=" * 72)

    results = compare(lengths, args.concurrency, args.step_ms)
    for result in results.values():
        print("  " + result.summary())

    sync = results["sync_batch"]
    async_ = results["async_refill"]
    speedup = sync.wall_ms / async_.wall_ms if async_.wall_ms else 0.0
    print()
    print(f"  wall-clock speedup from refilling slots : {speedup:.2f}x")
    print(f"  slot utilisation                        : {sync.slot_utilization:.1%} -> {async_.slot_utilization:.1%}")
    print(f"  idle slot time recovered                : {(sync.idle_slot_ms - async_.idle_slot_ms) / sync.idle_slot_ms:.1%}")

    print()
    print("=" * 72)
    print(f"round budget = {args.round_budget} steps")
    print("=" * 72)
    cut = simulate_truncation(lengths, args.round_budget, partial_rollout=False)
    partial = simulate_truncation(lengths, args.round_budget, partial_rollout=True)
    print("  " + cut.summary())
    print("  " + partial.summary())
    print()
    print(f"  episodes lost to truncation             : {cut.truncated} / {cut.n_episodes}")
    print(f"  compute wasted on them                  : {cut.wasted_steps} steps")
    print("  reward bias: truncated == failure, so the policy learns to avoid")
    print("  long-horizon tasks. Partial rollout removes that bias, not just the waste.")

    print()
    print("Note: a model, not a profiler. The ratio is what transfers; the")
    print("absolute seconds are only as good as --step-ms.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
