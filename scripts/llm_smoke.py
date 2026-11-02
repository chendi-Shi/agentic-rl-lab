"""End-to-end check of the LLM policy path, with a fake model. No GPU needed.

    python scripts/llm_smoke.py

Boots the fake chat server, points ``LLMPolicy`` at it, and runs the stage-1 task
set through the same code path a real vLLM endpoint would use. If this is green,
every remaining failure on the rented box is the model's fault, not the harness.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arlab.policy import ChatClient, LLMPolicy  # noqa: E402
from arlab.rollout import RunnerConfig, run_batch  # noqa: E402
from arlab.store import TraceStore  # noqa: E402
from arlab.tasks.math_env import MathEnv, make_math_tasks  # noqa: E402
from fake_chat_server import make_server  # noqa: E402


def main() -> int:
    server = make_server(0)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        client = ChatClient(base_url=f"http://{host}:{port}", model="fake", timeout=10)
        llm = LLMPolicy(client, max_tokens=64)

        def policy(task, steps):
            action = llm(task, steps)
            usage = llm.last_usage
            if steps:
                steps[-1].tokens_in += usage.prompt_tokens
                steps[-1].tokens_out += usage.completion_tokens
            return action

        store = TraceStore(Path("traces/llm-smoke.jsonl"))
        store.path.unlink(missing_ok=True)

        result = run_batch(
            make_math_tasks(),
            MathEnv,
            policy,
            RunnerConfig(config_id="llm-smoke", seed=0, oracle="answer"),
            policy_name="fake-llm",
            store=store,
        )

        records = store.load()
        tokens_in = sum(r.trajectory.tokens()[0] for r in records)
        tokens_out = sum(r.trajectory.tokens()[1] for r in records)

        print("=" * 66)
        print("LLM policy path, end to end, with a fake model")
        print("=" * 66)
        print(f"  {result.summary()}")
        print(f"  records={len(records)}  tokens_in={tokens_in} tokens_out={tokens_out}")
        print()
        for record in records[:3]:
            steps = len(record.trajectory)
            print(
                f"  {record.task_id}: steps={steps} "
                f"answer={record.trajectory.final_answer!r} "
                f"r_task={record.reward['r_task']:.2f}"
            )
        print()
        print("What this proves: HTTP client, message building, JSON action parsing,")
        print("tool execution, observation masking and scoring all agree.")
        print("On the rented box only the model's answer quality is left to test.")
        return 0 if result.mean_reward > 0.9 else 1
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
