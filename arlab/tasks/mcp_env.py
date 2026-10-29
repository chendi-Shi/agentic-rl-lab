"""MCP tool server adapter.

Stage-2 tasks (incident diagnosis) live behind an MCP server -- the platform's
own ``mcp.py`` exposes ``knowledge_search`` / ``service_metrics`` /
``incident_changes``. This adapter speaks plain JSON-RPC over a line transport
so it works with any MCP server, not just that one.

Note on rewards: an MCP server has no inspectable environment state, so these
tasks must use the *answer* oracle (``compute_structured_answer_reward``), not
the state oracle. ``snapshot()`` returns ``{}`` to make that explicit.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...

    def close(self) -> None: ...


class StdioTransport:
    """Line-delimited JSON-RPC over a child process's stdin/stdout."""

    def __init__(self, command: list[str], timeout: float = 30.0) -> None:
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._timeout = timeout
        self._next_id = 0

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("transport closed")
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params}
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._proc.stdin.flush()

        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server closed while waiting for {method}")
            message = json.loads(line)
            if message.get("id") != self._next_id:
                continue  # notification or out-of-order response
            if "error" in message:
                raise RuntimeError(f"MCP error: {message['error']}")
            return message.get("result") or {}

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()


class McpEnvironment:
    """``Environment`` implementation backed by an MCP tool server."""

    def __init__(self, transport: Transport) -> None:
        self._transport = transport
        self._tools: tuple[str, ...] | None = None

    def tool_names(self) -> tuple[str, ...]:
        if self._tools is None:
            result = self._transport.request("tools/list", {})
            self._tools = tuple(t["name"] for t in result.get("tools", []))
        return self._tools

    def tool_specs(self) -> list[dict[str, Any]]:
        return list(self._transport.request("tools/list", {}).get("tools", []))

    def call(self, name: str, args: dict[str, Any]) -> str:
        result = self._transport.request("tools/call", {"name": name, "arguments": dict(args)})
        if result.get("isError"):
            raise RuntimeError(_flatten(result) or "tool reported error")
        return _flatten(result)

    def snapshot(self) -> dict[str, Any]:
        """MCP exposes no state; answer-oracle tasks only."""

        return {}

    def close(self) -> None:
        self._transport.close()


def _flatten(result: dict[str, Any]) -> str:
    """Join MCP text content blocks into one observation string."""

    parts = [
        block.get("text", "")
        for block in result.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(parts)
