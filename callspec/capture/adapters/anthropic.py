"""Anthropic capture adapter: extract tool calls from Anthropic response dicts.

Anthropic returns tool calls as content blocks with type=="tool_use" inside
the response.content array. Each tool_use block contains id, name, and input
(a dict of arguments). Text blocks are interleaved with tool_use blocks.
"""

from __future__ import annotations

from typing import Any

from callspec.core.trajectory import ToolCall, ToolCallTrajectory


def extract_from_dict(response_dict: dict[str, Any]) -> ToolCallTrajectory:
    """Extract a ToolCallTrajectory from a raw Anthropic response dict."""
    model = response_dict.get("model", "")
    content_blocks = response_dict.get("content", [])

    calls: list[ToolCall] = []
    call_index = 0

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue

        calls.append(ToolCall(
            tool_name=block.get("name", ""),
            arguments=block.get("input", {}),
            call_index=call_index,
            model=model,
            provider="anthropic",
            call_id=block.get("id"),
        ))
        call_index += 1

    return ToolCallTrajectory(
        calls=calls,
        model=model,
        provider="anthropic",
        raw_response=response_dict,
    )
