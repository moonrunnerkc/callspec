"""OpenAI capture adapter: extract tool calls from OpenAI response dicts.

Handles three formats:
1. Chat Completions tool_calls: choices[0].message.tool_calls
2. Chat Completions legacy function_call: choices[0].message.function_call
3. Responses API (March 2025+): output items with type=="function_call"

Works on raw dict data, not the openai SDK response objects. The provider
adapter already calls model_dump() to produce the dict.
"""

from __future__ import annotations

import json
from typing import Any

from llm_assert.core.trajectory import ToolCall, ToolCallTrajectory


def extract_from_dict(response_dict: dict[str, Any]) -> ToolCallTrajectory:
    """Extract a ToolCallTrajectory from a raw OpenAI response dict."""
    model = response_dict.get("model", "")
    calls: list[ToolCall] = []
    call_index = 0

    # Chat Completions format
    choices = response_dict.get("choices", [])
    if choices:
        message = choices[0].get("message", {})

        # Current tool_calls format
        raw_tool_calls = message.get("tool_calls") or []
        for tc in raw_tool_calls:
            func = tc.get("function", {})
            arguments = _parse_arguments(func.get("arguments", "{}"))

            calls.append(ToolCall(
                tool_name=func.get("name", ""),
                arguments=arguments,
                call_index=call_index,
                model=model,
                provider="openai",
                call_id=tc.get("id"),
            ))
            call_index += 1

        # Legacy function_call format (deprecated but still in the wild)
        if not calls:
            function_call = message.get("function_call")
            if function_call:
                arguments = _parse_arguments(function_call.get("arguments", "{}"))
                calls.append(ToolCall(
                    tool_name=function_call.get("name", ""),
                    arguments=arguments,
                    call_index=0,
                    model=model,
                    provider="openai",
                ))

        return ToolCallTrajectory(
            calls=calls,
            model=model,
            provider="openai",
            raw_response=response_dict,
        )

    # Responses API format: output is a list of items
    output_items = response_dict.get("output", [])
    for item in output_items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            arguments = _parse_arguments(item.get("arguments", "{}"))
            calls.append(ToolCall(
                tool_name=item.get("name", ""),
                arguments=arguments,
                call_index=call_index,
                model=model,
                provider="openai",
                call_id=item.get("call_id"),
            ))
            call_index += 1

    return ToolCallTrajectory(
        calls=calls,
        model=model,
        provider="openai",
        raw_response=response_dict,
    )


def _parse_arguments(args_value: Any) -> dict[str, Any]:
    """Parse arguments from string or dict, handling malformed JSON gracefully."""
    if isinstance(args_value, dict):
        return args_value
    if isinstance(args_value, str):
        try:
            parsed = json.loads(args_value)
            if isinstance(parsed, dict):
                return parsed
            return {"_raw": parsed}
        except (json.JSONDecodeError, TypeError):
            return {"_raw": args_value}
    return {}
