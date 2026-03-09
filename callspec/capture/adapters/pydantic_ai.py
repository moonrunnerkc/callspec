"""Pydantic AI capture adapter: extract tool calls from ModelResponse objects.

Pydantic AI's ModelResponse has a .parts list that can contain ToolCallPart
objects with tool_name, args (dict), and tool_call_id attributes.
The adapter duck-types to avoid requiring pydantic-ai as a dependency.
"""

from __future__ import annotations

from typing import Any

from callspec.core.trajectory import ToolCall, ToolCallTrajectory


def extract_from_response(response: Any) -> ToolCallTrajectory:
    """Extract a ToolCallTrajectory from a Pydantic AI ModelResponse.

    Args:
        response: A Pydantic AI ModelResponse with a .parts attribute.
            Parts that are ToolCallPart instances have tool_name, args,
            and tool_call_id attributes.

    Returns:
        ToolCallTrajectory with all tool call parts extracted.
    """
    parts = getattr(response, "parts", []) or []
    model = getattr(response, "model_name", "") or ""

    calls: list[ToolCall] = []
    call_index = 0

    for part in parts:
        # Duck-type for ToolCallPart: has tool_name and args attributes
        if not (hasattr(part, "tool_name") and hasattr(part, "args")):
            continue

        calls.append(ToolCall(
            tool_name=getattr(part, "tool_name", ""),
            arguments=getattr(part, "args", {}),
            call_index=call_index,
            model=model,
            provider="pydantic_ai",
            call_id=getattr(part, "tool_call_id", None),
        ))
        call_index += 1

    return ToolCallTrajectory(
        calls=calls,
        model=model,
        provider="pydantic_ai",
    )
