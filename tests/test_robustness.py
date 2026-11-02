from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arlab.policy import INVALID_TOOL, ChatClient, LLMPolicy
from arlab.rollout import RunnerConfig, run_task
from arlab.tasks.math_env import MathEnv, make_math_tasks


class FlakyHandler(BaseHTTPRequestHandler):
    """Fails the first ``fail_count`` requests with the given status, then answers."""

    fail_count = 0
    status = 500
    reply = '{"tool": "python", "args": {"code": "1+1"}}'
    hits = 0

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        length = int(self.headers.get("content-length", 0))
        self.rfile.read(length)
        FlakyHandler.hits += 1

        if FlakyHandler.hits <= FlakyHandler.fail_count:
            body = b"upstream exploded"
            self.send_response(FlakyHandler.status)
        else:
            payload = {
                "choices": [{"message": {"content": FlakyHandler.reply}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:
        return


@pytest.fixture
def flaky():
    FlakyHandler.hits = 0
    FlakyHandler.fail_count = 0
    FlakyHandler.status = 500
    server = ThreadingHTTPServer(("127.0.0.1", 0), FlakyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_retries_transient_5xx_then_succeeds(flaky):
    FlakyHandler.fail_count = 2
    client = ChatClient(base_url=flaky, timeout=10, max_retries=3, backoff=0.01)

    text, usage = client.complete([{"role": "user", "content": "hi"}])

    assert "python" in text
    assert usage.prompt_tokens == 5
    assert FlakyHandler.hits == 3


def test_gives_up_after_retries(flaky):
    FlakyHandler.fail_count = 10
    client = ChatClient(base_url=flaky, timeout=10, max_retries=1, backoff=0.01)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        client.complete([{"role": "user", "content": "hi"}])


def test_does_not_retry_client_errors(flaky):
    FlakyHandler.fail_count = 10
    FlakyHandler.status = 400
    client = ChatClient(base_url=flaky, timeout=10, max_retries=3, backoff=0.01)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        client.complete([{"role": "user", "content": "hi"}])
    assert FlakyHandler.hits == 1


def test_policy_turns_a_dead_endpoint_into_one_invalid_step(flaky):
    """A broken endpoint must cost a step, not the whole run."""
    FlakyHandler.fail_count = 100
    policy = LLMPolicy(ChatClient(base_url=flaky, timeout=5, max_retries=0, backoff=0.01))

    action = policy(make_math_tasks()[0], [])

    assert action.name == INVALID_TOOL
    assert "HTTP 500" in action.args["error"]


def test_run_survives_a_model_that_only_emits_garbage(tmp_path):
    """Unparseable output becomes invalid calls until the step budget is spent."""
    from arlab.store import TraceStore

    FlakyHandler.fail_count = 0
    FlakyHandler.reply = "I am not going to call any tool."
    server = ThreadingHTTPServer(("127.0.0.1", 0), FlakyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        policy = LLMPolicy(ChatClient(base_url=f"http://127.0.0.1:{server.server_address[1]}", timeout=10))
        task = make_math_tasks()[0]
        store = TraceStore(tmp_path / "t.jsonl")

        record = run_task(
            task,
            MathEnv,
            policy,
            RunnerConfig(config_id="c", seed=0, oracle="answer"),
            policy_name="garbage",
            store=store,
        )

        assert record.reward["r_task"] == pytest.approx(0.0)
        assert record.reward["p_invalid"] > 0
        assert len(record.trajectory) <= task.max_steps + 1
    finally:
        server.shutdown()
        server.server_close()
