from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from fake_chat_server import make_server  # noqa: E402
from arlab.policy import ChatClient, LLMPolicy  # noqa: E402
from arlab.rollout import RunnerConfig, run_batch  # noqa: E402
from arlab.store import TraceStore  # noqa: E402
from arlab.tasks.math_env import MathEnv, make_math_tasks  # noqa: E402


@pytest.fixture
def fake_endpoint():
    server = make_server(0)
    host, port = server.server_address[:2]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()


def _policy(base_url: str) -> LLMPolicy:
    return LLMPolicy(ChatClient(base_url=base_url, model="fake", timeout=10), max_tokens=64)


def test_llm_policy_solves_tasks_through_http(fake_endpoint, tmp_path):
    """The whole LLM path: HTTP -> messages -> parse -> tool -> score."""

    llm = _policy(fake_endpoint)
    store = TraceStore(tmp_path / "llm.jsonl")

    result = run_batch(
        make_math_tasks()[:3],
        MathEnv,
        llm,
        RunnerConfig(config_id="t", seed=0, oracle="answer"),
        policy_name="fake-llm",
        store=store,
    )

    assert result.n_run == 3
    assert result.mean_reward > 0.9
    records = store.load()
    assert all(record.reward["r_task"] == pytest.approx(1.0) for record in records)


def test_llm_policy_reads_multiple_turns(fake_endpoint):
    """The second call must see the observation from the first, not the task text."""

    llm = _policy(fake_endpoint)
    task = make_math_tasks()[0]

    first = llm(task, [])
    assert first.name == "python"

    from arlab.tasks.schema import Step

    # The observation is the raw tool output; build_messages adds the prefix.
    steps = [Step(tool_call=first, observation="400")]
    second = llm(task, steps)
    assert getattr(second, "text", None) == "400"


def test_unknown_instruction_degrades_gracefully(fake_endpoint):
    """A task the fake model has no plan for must not hang or crash the loop."""

    from arlab.tasks.schema import Task

    llm = _policy(fake_endpoint)
    task = Task(task_id="x", instruction="unscripted", required_fields={"answer": "1"}, tools=("python",))

    action = llm(task, [])
    assert getattr(action, "text", None) == "unknown task"
