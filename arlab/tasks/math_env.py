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
import math
import re
import statistics
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
    # math helpers, injected so `import math` is never needed
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "exp": math.exp,
    "median": statistics.median,
    "print": lambda *args: " ".join(str(a) for a in args),
}
_FORBIDDEN = (ast.Attribute, ast.Subscript, ast.Lambda, ast.Import, ast.ImportFrom)


class ArithmeticExecutor:
    """Runs one expression, or a short block of statements, with no I/O."""

    max_code_chars = 2000
    _IMPORT_MATH = re.compile(r"^\s*(?:import\s+math|from\s+math\s+import\s+[^\n]*?)\s*$", re.MULTILINE)
    _MATH_ATTR = re.compile(r"\bmath\.([A-Za-z_]\w*)")

    def _preprocess(self, code: str) -> str:
        """Let ``import math`` and ``math.sqrt(...)`` through.

        Attribute access is forbidden because it is the classic sandbox escape,
        but models reach for ``math.sqrt`` constantly. Dropping the import and
        rewriting the call onto the injected helper is better than failing the
        whole answer -- which is exactly what happened on the hard tier.
        """

        code = self._IMPORT_MATH.sub("", code)
        return self._MATH_ATTR.sub(r"\1", code)

    def run(self, code: str) -> str:
        code = (code or "").strip()
        if not code:
            raise ValueError("empty code")
        if len(code) > self.max_code_chars:
            raise ValueError("code too long")
        code = self._preprocess(code)
        if not code.strip():
            raise ValueError("empty code after preprocessing")

        # Models write both `17*23+9` and `x = 17*23+9 \\n print(x)`. Accept a
        # module, run the body, and return the last expression's value or
        # whatever was printed -- statements are the common case, so treating
        # them as a syntax error would fail most real replies.
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = ast.parse(code, mode="eval")

        self._validate(tree)
        namespace = {"__builtins__": {}, "pi": math.pi, "e": math.e}
        namespace.update(_ALLOWED_CALLS)

        printed: list[str] = []

        def _print(*args, **kwargs) -> None:
            printed.append(" ".join(str(a) for a in args))

        namespace["print"] = _print

        body = getattr(tree, "body", [])
        value = None
        if isinstance(tree, ast.Expression):
            value = eval(compile(tree, "<tool>", "eval"), namespace, namespace)
        elif body and isinstance(body[-1], ast.Expr):
            head = ast.Module(body=body[:-1], type_ignores=[])
            if body[:-1]:
                exec(compile(head, "<tool>", "exec"), namespace, namespace)
            value = eval(compile(ast.Expression(body[-1].value), "<tool>", "eval"), namespace, namespace)
        else:
            exec(compile(tree, "<tool>", "exec"), namespace, namespace)

        if printed:
            return "\n".join(printed)
        return "" if value is None else str(value)

    def _validate(self, tree: ast.AST) -> None:
        """Block the escape hatches, allow ordinary code.

        Attribute access and subscripts are the classic sandbox escapes
        (``().__class__``), so they stay forbidden. Plain assignments and local
        names are allowed: real models write ``x = ...`` far more often than a
        bare expression, and rejecting that turns every task into a syntax error.
        """

        for node in ast.walk(tree):
            if isinstance(node, _FORBIDDEN):
                raise ValueError(f"forbidden construct: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.startswith("__"):
                raise ValueError(f"dunder name: {node.id}")
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                raise ValueError(f"forbidden construct: {type(node).__name__}")
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
