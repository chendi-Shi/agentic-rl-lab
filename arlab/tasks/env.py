"""Environment protocol plus a deterministic synthetic business backend.

The synthetic backend exists so the whole pipeline (task -> rollout ->
reward -> diagnostics) can be exercised on CPU in tests. It is a *test double*:
the real stage-2 tasks plug into ``byte-agent-platform``'s MCP tools through the
same three-method interface.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable


@runtime_checkable
class Environment(Protocol):
    """Anything the agent can call, with inspectable final state."""

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        """Execute a tool call, return an observation string."""

    def snapshot(self) -> dict[str, Any]:
        """Return the verifiable state used by the reward oracle."""

    def tool_names(self) -> tuple[str, ...]:
        """Names accepted by :meth:`call`."""


class SyntheticBusinessEnv:
    """Three services over an in-memory order table. Fully deterministic."""

    ORDERS: dict[str, dict[str, Any]] = {
        "A123": {"status": "paid", "amount": 199.0, "refunded": False, "refund_amount": 0.0},
        "B456": {"status": "shipped", "amount": 89.5, "refunded": False, "refund_amount": 0.0},
        "C789": {"status": "delivered", "amount": 320.0, "refunded": False, "refund_amount": 0.0},
    }
    TOOLS = ("orders.get", "orders.refund", "shipping.track")

    def __init__(self) -> None:
        import copy

        self._orders = copy.deepcopy(self.ORDERS)
        # ``last_queried`` starts False on purpose: it lets a task require
        # *evidence of inspection*, so "do nothing and claim success" cannot
        # satisfy a task whose end state is already correct.
        for order in self._orders.values():
            order["last_queried"] = False

    def tool_names(self) -> tuple[str, ...]:
        return self.TOOLS

    def call(self, name: str, args: Mapping[str, Any]) -> str:
        if name not in self.TOOLS:
            raise KeyError(f"unknown tool: {name}")
        order_id = str(args.get("order_id", ""))
        if order_id not in self._orders:
            raise KeyError(f"unknown order: {order_id!r}")
        order = self._orders[order_id]

        if name == "orders.get":
            order["last_queried"] = True
            return (
                f"order={order_id} status={order['status']} amount={order['amount']} "
                f"refunded={order['refunded']}"
            )
        if name == "shipping.track":
            order["last_queried"] = True
            carrier = {"shipped": "SF", "delivered": "JD"}.get(order["status"], "N/A")
            return f"order={order_id} carrier={carrier} status={order['status']}"
        if order["refunded"]:
            return f"order={order_id} already refunded"
        order["refunded"] = True
        order["refund_amount"] = order["amount"]
        amount = order["refund_amount"]
        return f"order={order_id} refunded amount={amount}"

    def snapshot(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for order_id, order in self._orders.items():
            for key, value in order.items():
                state[f"{order_id}.{key}"] = value
        return state
