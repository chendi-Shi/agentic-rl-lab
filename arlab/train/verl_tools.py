"""Register our lookup environment as verl tools.

Only importable where verl is installed (the GPU box). verl infers the OpenAI
tool schema from the function itself: it needs a Google-style docstring with an
``Args:`` block and a type hint per parameter, otherwise registration raises
``DocstringParsingException``.

Both tools are read-only and stateless, which is why one shared environment is
safe: ``orders.get`` never mutates the table and ``calc`` is pure. A task that
needed ``orders.refund`` would require verl's stateful ``BaseTool`` instead.
"""

from __future__ import annotations

from verl.tools.function_tool import function_tool

from arlab.tasks.lookup_tasks import LookupEnv

_ENV = LookupEnv()


@function_tool("orders.get")
def orders_get(order_id: str) -> str:
    """Look up one order and return its status, amount and refund state.

    Args:
        order_id: Order identifier, for example A123.

    Returns:
        A single-line observation such as "order=A123 status=paid amount=199.0".
    """

    return _ENV.call("orders.get", {"order_id": order_id})


@function_tool("calc")
def calc(expression: str) -> str:
    """Evaluate one arithmetic expression and return its exact value.

    Args:
        expression: A Python arithmetic expression, for example 199.0*7.5/10.

    Returns:
        The value as text, for example "149.25".
    """

    return _ENV.call("calc", {"expression": expression})


TOOL_NAMES = ("orders.get", "calc")
