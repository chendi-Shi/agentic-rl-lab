"""Quantify the partial-rollout trade on real episodes.

Run this on the box with a vLLM server up. It answers the only question that
matters for the integration: how much step latency does truncation save, and how
much reward does it cost?

    python scripts/bench_partial.py --task agg-sum-all --group 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arlab.policy.openai import ChatClient, LLMPolicy, system_prompt
from arlab.rollout.deadline import collect_with_deadline
from arlab.rollout.runner import RunnerConfig
from arlab.tasks.lookup_tasks import LookupEnv
from arlab.tasks.stage_io import load_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("tasks/stage3-hard.json"))
    parser.add_argument("--task", default="agg-sum-all")
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--base-url", default="http://127.0.0.1:6006")
    parser.add_argument("--model", default="qwen3-4b")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--out", type=Path, default=Path("reports/partial_bench.json"))
    args = parser.parse_args()

    loaded = load_tasks(args.tasks)
    task = next((item.task for item in loaded if item.task.task_id == args.task), None)
    if task is None:
        raise SystemExit(f"{args.task} not in {args.tasks}")

    def policy_factory(seed: int):
        client = ChatClient(
            base_url=args.base_url,
            model=args.model,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}, "seed": seed},
        )
        llm = LLMPolicy(client, max_tokens=512, temperature=args.temperature)
        return lambda task, steps: llm(task, steps)

    cfg = RunnerConfig(config_id="partial", seed=11, oracle="answer")

    # First pass with no deadline: gives the natural p50/p99 of this task, which is
    # where a sensible deadline comes from. Guessing a constant would either never
    # fire or cut everything.
    _, baseline = collect_with_deadline(
        task, LookupEnv, policy_factory, cfg, group_size=args.group, workers=args.group
    )
    print(f"no deadline : {baseline.summary()}")

    deadline = baseline.percentile(0.5) * 1.5
    _, cut = collect_with_deadline(
        task, LookupEnv, policy_factory, cfg,
        group_size=args.group, workers=args.group, deadline_s=deadline,
    )
    print(f"deadline {deadline:.1f}s : {cut.summary()}")

    rows = [
        {"mode": "no_deadline", **baseline.__dict__, "deadline_s": None},
        {
            "mode": "deadline",
            "deadline_s": round(deadline, 2),
            **{k: v for k, v in cut.__dict__.items() if k != "durations"},
        },
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    saved = baseline.wall_seconds - cut.wall_seconds
    print(
        f"\nwall saved: {saved:+.1f}s "
        f"({saved / baseline.wall_seconds * 100:+.0f}%)  "
        f"reward cost: {cut.solved - baseline.solved:+d} solved"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
