"""A fake Claude client and response builders for tests.

The fake client replays a scripted list of responses, so the agent loop can
be tested exactly, offline, and for free.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(tool_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def fake_response(*blocks: SimpleNamespace, stop_reason: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(
        content=list(blocks),
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )


class FakeMessagesClient:
    """Stands in for anthropic.Anthropic().messages."""

    def __init__(self, scripted_responses: list[SimpleNamespace]) -> None:
        self.scripted_responses = list(scripted_responses)
        self.received_calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        # Snapshot the message list, since the agent keeps appending to it.
        self.received_calls.append({**kwargs, "messages": list(kwargs["messages"])})
        if not self.scripted_responses:
            raise AssertionError("Agent made more model calls than the test scripted.")
        return self.scripted_responses.pop(0)
