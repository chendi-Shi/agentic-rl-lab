"""Stage-1 task family: arithmetic solved through a code-interpreter tool.

Why this exists: the roadmap requires proving the RL loop converges on a task
with *known* public evidence before touching the bespoke diagnosis task. Math
with a calculator is the cheapest such task -- exact-match reward, sub-millisecond
tools, no network, no GPU needed to test the harness.

Two executors are provided:

* ``ArithmeticExecutor`` -- AST-validated, dependency-free, used in tests and
  offline development. Rejects attribute access, imports and unknown names.
* ``SubprocessExecutor`` -- runs real Python in a child process with a timeout.
  Use this in production, inside the existing Docker sandbox; never on the host.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, runtime_checkable

from arlab.tasks.schema import Task


@runtime_checkable
class Executor(Protocol):
    def run(self, code: str) -> str: ...


_ALLOWED_NAMES = {"abs", "min", "max", "round", "sum", "len", "pow", "int", "float", "str", "range"}
_ALLOWED_CALLS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sum": sum,
    "len": len,
    "pow": pow,
    "int": int,
    "float": float,
    "str": str,
    "range": range,
    "print": lambda *args: " ".join(str(a) for a in args),
}
_FORBIDDEN = (ast.Attribute, ast.Subscript, ast.Lambda, ast.Import, ast.ImportFrom)


class ArithmeticExecutor:
    """Evaluates a single arithmetic expression, no I/O, no imports."""

    max_code_chars = 2000

    def run(self, code: str) -> str:
        code = (code or "").strip()
        if not code:
            raise ValueError("empty code")
        if len(code) > self.max_code_chars:
            raise ValueError("code too long")
        tree = ast.parse(code, mode="eval")
        self._validate(tree)
        value = eval(compile(tree, "<tool>", "eval"), {"__builtins__": {}}, dict(_ALLOWED_CALLS))
        return str(value)

    def _validate(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN):
                raise ValueError(f"forbidden construct: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id not in _ALLOWED_NAMES | {"print"}:
                raise ValueError(f"unknown name: {node.id}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in _ALLOWED_CALLS:
                    raise ValueError(f"unknown function: {node.func.id}")


class SubprocessExecutor:
    """Real Python in a child process. Requires the Docker sandbox in production."""

    def __init__(self, timeout: float = 10.0, max_output: int = 4000) -> None:
        self.timeout = timeout
        self.max_output = max_output

    def run(self, code: str) -> str:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"code timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or "nonzero exit").strip()[: self.max_output])
        return (proc.stdout or "").strip()[: self.max_output]


class MathEnv:
    """Environment exposing one tool: ``python(code)``."""

    TOOLS = ("python",)

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor or ArithmeticExecutor()
        self._calls = 0

    def tool_names(self) -> tuple[str, ...]:
        return self.TOOLS

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        if name != "python":
            raise KeyError(f"unknown tool: {name}")
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("python tool requires a non-empty 'code' string")
        self._calls += 1
        return self._executor.run(code)

    def snapshot(self) -> dict[str, Any]:
        return {"python_calls": self._calls}


def make_math_tasks() -> list[Task]:
    """Six deterministic tasks; answers are exact strings for the answer oracle."""

    raw = [
        ("m-01", "17 * 23 + 9 等于多少？用 python 算，然后提交答案。", "400"),
        ("m-02", "(144 / 12) ** 2 - 19 等于多少？", "125"),
        ("m-03", "从 1 加到 50 的和是多少？", "1275"),
        ("m-04", "2 的 10 次方减去 24 是多少？", "1000"),
        ("m-05", "一个 12 边多边形内角和是多少度？", "1800"),
        ("m-06", "6! 加 5! 等于多少？", "840"),
    ]
    return [
        Task(
            task_id=task_id,
            instruction=instruction,
            required_fields={"answer": answer},
            max_steps=6,
            tools=MathEnv.TOOLS,
            family="math",
        )
        for task_id, instruction, answer in raw
    ]
