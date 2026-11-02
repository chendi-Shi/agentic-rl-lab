"""OpenAI-compatible fake chat server that plays a tool-using agent.

It exists so the whole ``--policy llm`` path can be exercised on CPU: HTTP
request, message building, action parsing, tool execution, observation masking
and scoring. On the rented box the only variable left is then the model's actual
quality, not the plumbing.

The scripted behaviour is trivial by design: call the code interpreter once, then
echo the observation back as the final answer. If that scores well, the loop is
correct.

    python scripts/fake_chat_server.py --port 8000     # then point a client at it
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arlab.tasks.math_env import make_math_tasks  # noqa: E402
from arlab.tasks.scripted import MATH_CODE  # noqa: E402

OBS_PREFIX = "observation: "


def build_plans() -> dict[str, str]:
    """instruction -> code to run."""

    return {
        task.instruction: MATH_CODE[task.task_id]
        for task in make_math_tasks()
        if task.task_id in MATH_CODE
    }


PLANS = build_plans()


def decide(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the next action from the conversation so far."""

    observations = [m.get("content", "") for m in messages if m.get("content", "").startswith(OBS_PREFIX)]
    if observations:
        value = observations[-1][len(OBS_PREFIX) :].strip()
        return {"final_answer": value}

    instruction = messages[1].get("content", "") if len(messages) > 1 else ""
    code = PLANS.get(instruction)
    if code is None:
        return {"final_answer": "unknown task"}
    return {"tool": "python", "args": {"code": code}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib hook name
        length = int(self.headers.get("content-length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "bad json"})
            return

        messages = payload.get("messages") or []
        action = decide(messages)
        body = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "model": payload.get("model", "fake"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(action, ensure_ascii=False)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 32, "completion_tokens": 12, "total_tokens": 44},
        }
        self._send(200, body)

    def _send(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        return


def make_server(port: int = 0) -> ThreadingHTTPServer:
    """Bind on 127.0.0.1; pass port 0 to get a free one."""

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = make_server(args.port)
    host, port = server.server_address[:2]
    print(f"fake chat server on http://{host}:{port}  ({len(PLANS)} scripted tasks)")
    print('point a client at it:  --base-url http://127.0.0.1:%d --model fake' % port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
