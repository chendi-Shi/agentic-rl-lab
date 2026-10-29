"""Run a frozen task set and record it.

    # pipeline self-check, no model, no GPU
    python scripts/run_baseline.py --tasks tasks/stage1.json --policy scripted --oracle answer

    # real baseline against a local vLLM server
    python scripts/run_baseline.py --tasks tasks/stage1.json \
        --policy llm --base-url http://127.0.0.1:8000 --model qwen3-8b --oracle answer

Re-running is free: completed ``(task, policy, config, seed)`` keys are skipped.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.diagnostics import (  # noqa: E402
    classify_difficulty,
    detect_advantage_collapse,
    length_percentiles,
)
from arlab.policy import ChatClient, LLMPolicy, annotate_tokens  # noqa: E402
from arlab.rollout import RunnerConfig, run_batch  # noqa: E402
from arlab.store import TraceStore  # noqa: E402
from arlab.tasks.stage_io import LoadedTask, load_tasks  # noqa: E402
from arlab.tasks.scripted import scripted_policy_for  # noqa: E402


class _SharedEnv:
    """One MCP server, many tasks: the transport outlives a single episode."""

    def __init__(self, env) -> None:
        self._env = env

    def tool_names(self):
        return self._env.tool_names()

    def call(self, name, args):
        return self._env.call(name, args)

    def snapshot(self):
        return {}


def make_env_factory(loaded: list[LoadedTask], env_kind: str, mcp_command: str):
    family = loaded[0].task.family if loaded else "math"
    if env_kind == "auto":
        env_kind = "math" if family == "math" else ("business" if family in {"business", "default"} else "mcp")

    if env_kind == "math":
        from arlab.tasks.math_env import MathEnv

        return MathEnv, "answer"

    if env_kind == "business":
        from arlab.tasks.env import SyntheticBusinessEnv

        return SyntheticBusinessEnv, "state"

    from arlab.tasks.mcp_env import McpEnvironment, StdioTransport

    transport = StdioTransport(shlex.split(mcp_command))
    shared = _SharedEnv(McpEnvironment(transport))
    return lambda: shared, "answer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--store", type=Path, default=Path("traces/baseline.jsonl"))
    parser.add_argument("--policy", choices=["scripted", "llm"], default="scripted")
    parser.add_argument("--env", choices=["auto", "math", "business", "mcp"], default="auto")
    parser.add_argument(
        "--mcp-command",
        default=f'"{sys.executable}" scripts/fake_mcp_server.py',
        help="command that starts the MCP server (used when --env mcp)",
    )
    parser.add_argument("--oracle", choices=["answer", "state"], default=None)
    parser.add_argument("--config-id", default="baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--split", default=None, help="only run this split (dev / holdout)")
    parser.add_argument("--limit", type=int, default=0, help="run at most N tasks")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    loaded = load_tasks(args.tasks, split=args.split)
    if args.limit:
        loaded = loaded[: args.limit]
    if not loaded:
        raise SystemExit("no tasks matched")

    env_factory, default_oracle = make_env_factory(loaded, args.env, args.mcp_command)
    oracle = args.oracle or default_oracle
    cfg = RunnerConfig(config_id=args.config_id, seed=args.seed, oracle=oracle)

    if args.policy == "scripted":
        policy = scripted_policy_for
        policy_name = "scripted"
    else:
        client = ChatClient(base_url=args.base_url, model=args.model)
        llm = LLMPolicy(client, max_tokens=args.max_tokens)
        policy_name = f"llm:{args.model}"

        def policy(task, steps):  # noqa: ANN001 - thin wrapper to record token usage
            action = llm(task, steps)
            usage = llm.last_usage
            if steps:
                steps[-1].tokens_in += usage.prompt_tokens
                steps[-1].tokens_out += usage.completion_tokens
            return action

    store = TraceStore(args.store)
    started = time.perf_counter()
    tasks = [item.task for item in loaded]
    result = run_batch(tasks, env_factory, policy, cfg, policy_name=policy_name, store=store)
    wall = time.perf_counter() - started

    records = [r for r in store.load() if r.policy == policy_name and r.config_id == args.config_id]
    rewards = store.rewards_by_task()
    tokens_in = sum(r.trajectory.tokens()[0] for r in records)
    tokens_out = sum(r.trajectory.tokens()[1] for r in records)

    print("=" * 68)
    print(f"policy={policy_name}  oracle={oracle}  store={args.store}")
    print(f"{result.summary()}")
    print(f"tokens: in={tokens_in} out={tokens_out}  records={len(records)}")

    collapse = detect_advantage_collapse(rewards)
    diff = classify_difficulty(rewards)
    tail = length_percentiles([len(r.trajectory) for r in records])
    print(collapse.summary())
    print(f"keep {len(diff.kept)} / {len(diff.stats)} tasks in the 20-80% band")
    print(f"length: p50={tail['p50']:.0f} p90={tail['p90']:.0f} p99={tail['p99']:.0f}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "policy": policy_name,
        "oracle": oracle,
        "config_id": args.config_id,
        "seed": args.seed,
        "tasks_file": str(args.tasks),
        "run": result.n_run,
        "skipped": result.n_skipped,
        "mean_reward": result.mean_reward,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "wall_seconds": round(wall, 2),
        "collapse": {"n_groups": collapse.n_groups, "frac_collapsed": collapse.frac_collapsed},
        "difficulty": {"kept": diff.kept, "too_easy": diff.too_easy, "too_hard": diff.too_hard},
        "length": tail,
    }
    out = args.reports_dir / f"baseline-{policy_name.replace(':', '-')}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written: {out}")

    if args.policy == "scripted":
        print()
        print("NOTE: the scripted policy replays the correct actions, so a perfect")
        print("score here only proves the pipeline works. It is not a baseline and")
        print("must not be reported as a model result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
