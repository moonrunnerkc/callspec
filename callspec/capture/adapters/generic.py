"""Generic capture adapter: accept raw tool-call dicts from any source.

For frameworks not explicitly supported, users pass a plain list of dicts
with "name" and "arguments" keys. This is the fallback adapter.
"""

from __future__ import annotations

from typing import Any

from callspec.core.trajectory import ToolCall, ToolCallTrajectory


def extract_from_list(
    tool_call_dicts: list[dict[str, Any]],
    model: str = "",
    provider: str = "",
) -> ToolCallTrajectory:
    """Build a ToolCallTrajectory from a plain list of tool-call dicts.

    Args:
        tool_call_dicts: List of dicts, each with at minimum "name" and
            "arguments" keys. Also accepts "tool_name" and "args" as aliases.
        model: Optional model identifier to attach to the trajectory.
        provider: Optional provider name to attach to the trajectory.

    Returns:
        ToolCallTrajectory with the calls indexed sequentially.
    """
    calls: list[ToolCall] = []

    for index, tc_dict in enumerate(tool_call_dicts):
        calls.append(ToolCall(
            tool_name=tc_dict.get("name", tc_dict.get("tool_name", "")),
            arguments=tc_dict.get("arguments", tc_dict.get("args", {})),
            call_index=index,
            model=model,
            provider=provider,
            call_id=tc_dict.get("id"),
        ))

    return ToolCallTrajectory(
        calls=calls,
        model=model,
        provider=provider,
    )
