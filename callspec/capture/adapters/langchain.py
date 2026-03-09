"""LangChain capture adapter: extract tool calls from AIMessage objects.

LangChain's AIMessage (langchain-core >= 0.2) exposes tool_calls as a list
of dicts with keys: name, args, id, type. The adapter duck-types the message
to avoid requiring langchain as a dependency.
"""

from __future__ import annotations

from typing import Any

from callspec.core.trajectory import ToolCall, ToolCallTrajectory


def extract_from_message(message: Any) -> ToolCallTrajectory:
    """Extract a ToolCallTrajectory from a LangChain AIMessage.

    Args:
        message: A LangChain AIMessage with a .tool_calls attribute.
            Each tool call is a dict with "name", "args", and optionally "id".

    Returns:
        ToolCallTrajectory with all tool calls from the message.
    """
    raw_tool_calls = getattr(message, "tool_calls", []) or []
    model = getattr(message, "response_metadata", {}).get("model_name", "")

    calls: list[ToolCall] = []

    for index, tc in enumerate(raw_tool_calls):
        calls.append(ToolCall(
            tool_name=tc.get("name", ""),
            arguments=tc.get("args", {}),
            call_index=index,
            model=model,
            provider="langchain",
            call_id=tc.get("id"),
        ))

    return ToolCallTrajectory(
        calls=calls,
        model=model,
        provider="langchain",
    )
