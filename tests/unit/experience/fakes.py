"""A scripted stand-in for the Anthropic client: replies in order, records each request."""

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def tool_use(name: str, arguments: dict[str, Any], call_id: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=arguments)


def response(*blocks: SimpleNamespace, stop_reason: str | None = None) -> SimpleNamespace:
    calls = any(b.type == "tool_use" for b in blocks)
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason or ("tool_use" if calls else "end_turn")
    )


def api_error() -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.invalid"))


class FakeClient:
    def __init__(self, *replies: SimpleNamespace | Exception | Callable[[], Any]) -> None:
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        # Copy the messages: the coach appends to the same list after the call
        self.requests.append({**request, "messages": list(request["messages"])})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
