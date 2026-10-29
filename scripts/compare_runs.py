"""Paired comparison of two runs, with a confidence interval.

    python scripts/compare_runs.py --a traces/sft.jsonl --b traces/rl.jsonl

Pairs by task id, so task difficulty cancels out; reports the mean difference and
a bootstrap CI. A difference without a CI that excludes zero is not a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.eval import paired_bootstrap  # noqa: E402
from arlab.store import TraceStore  # noqa: E402


def per_task_means(path: Path, config_id: str | None = None, policy: str | None = None) -> dict[str, float]:
    totals: dict[str, list[float]] = {}
    for record in TraceStore(path).load():
        if config_id is not None and record.config_id != config_id:
            continue
        if policy is not None and record.policy != policy:
            continue
        totals.setdefault(record.task_id, []).append(float(record.reward.get("total", 0.0)))
    return {task: sum(values) / len(values) for task, values in totals.items()}


def token_totals(path: Path) -> tuple[int, int]:
    tin = tout = 0
    for record in TraceStore(path).load():
        a, b = record.trajectory.tokens()
        tin += a
        tout += b
    return tin, tout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True, help="baseline store")
    parser.add_argument("--b", type=Path, required=True, help="candidate store")
    parser.add_argument("--config-a", default=None)
    parser.add_argument("--config-b", default=None)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/compare.json"))
    args = parser.parse_args()

    means_a = per_task_means(args.a, args.config_a)
    means_b = per_task_means(args.b, args.config_b)
    shared = sorted(set(means_a) & set(means_b))
    if not shared:
        raise SystemExit("no overlapping task ids between the two stores")

    a = [means_a[t] for t in shared]
    b = [means_b[t] for t in shared]
    result = paired_bootstrap(a, b, n_boot=args.n_boot, seed=args.seed)

    print("=" * 72)
    print(f"{args.label_a}: {args.a}   ({len(means_a)} tasks)")
    print(f"{args.label_b}: {args.b}   ({len(means_b)} tasks)")
    print(f"paired on {len(shared)} shared tasks")
    print("=" * 72)
    print(f"  mean {args.label_a:<12}{result.mean_a:.4f}")
    print(f"  mean {args.label_b:<12}{result.mean_b:.4f}")
    print(f"  difference      {result.mean_diff:+.4f}   "
          f"95% CI [{result.ci_low:+.4f}, {result.ci_high:+.4f}]")
    print(f"  P({args.label_b}>{args.label_a})     {result.prob_b_better:.3f}")
    print(f"  verdict         {'SIGNIFICANT' if result.significant else 'NOT significant'}")

    tin_a, tout_a = token_totals(args.a)
    tin_b, tout_b = token_totals(args.b)
    if tout_a and tout_b:
        print()
        print(f"  tokens out  {args.label_a}={tout_a}  {args.label_b}={tout_b}  "
              f"({tout_b / tout_a:.2f}x)")
        print(f"  tokens in   {args.label_a}={tin_a}  {args.label_b}={tin_b}  "
              f"({tin_b / tin_a if tin_a else float('nan'):.2f}x)")

    print()
    print("  Report the CI, not the point estimate. If it straddles zero, say")
    print("  'no significant difference' -- that is a finding, not a failure.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "label_a": args.label_a,
                "label_b": args.label_b,
                "store_a": str(args.a),
                "store_b": str(args.b),
                "n_shared_tasks": len(shared),
                "mean_a": result.mean_a,
                "mean_b": result.mean_b,
                "diff": result.mean_diff,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "prob_b_better": result.prob_b_better,
                "significant": result.significant,
                "tokens_out": {"a": tout_a, "b": tout_b},
                "per_task": {t: {"a": means_a[t], "b": means_b[t]} for t in shared},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
