"""Tool-call trajectory data model.

The entire assertion and snapshot system operates on ToolCallTrajectory,
an ordered list of ToolCall records. Every capture adapter, normalizer,
assertion, and diff operation works against this structure.

A ToolCall is a frozen record of a single tool invocation. A
ToolCallTrajectory is the ordered sequence of all tool calls from
one LLM response (or one agent turn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation extracted from an LLM response.

    Attributes:
        tool_name: The function/tool name the agent invoked.
        arguments: The JSON parameters passed to the tool.
        call_index: Position in the trajectory sequence (0-based).
        model: Which model produced this call.
        provider: Which provider adapter captured it.
        timestamp: When the call was captured (optional).
        raw: Original provider response fragment for debugging.
        call_id: Provider-assigned identifier for this call (optional).
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_index: int = 0
    model: str = ""
    provider: str = ""
    timestamp: datetime | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON storage and comparison."""
        result: dict[str, Any] = {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_index": self.call_index,
        }
        if self.model:
            result["model"] = self.model
        if self.provider:
            result["provider"] = self.provider
        if self.timestamp is not None:
            result["timestamp"] = self.timestamp.isoformat()
        if self.call_id is not None:
            result["call_id"] = self.call_id
        return result


@dataclass
class ToolCallTrajectory:
    """An ordered sequence of tool calls from one LLM response or agent turn.

    The trajectory is the unit of comparison for assertions, snapshots,
    and drift detection. Two trajectories are equal when their tool_names
    and arguments match in order; metadata fields (model, provider,
    timestamps, call IDs) are excluded from equality checks since they
    vary across runs.

    Attributes:
        calls: Ordered list of ToolCall records.
        model: The model that produced this trajectory.
        provider: The provider that captured it.
        raw_response: The full provider response for debugging.
    """

    calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_names(self) -> list[str]:
        """The ordered list of tool names in this trajectory."""
        return [call.tool_name for call in self.calls]

    @property
    def is_empty(self) -> bool:
        return len(self.calls) == 0

    def calls_to(self, tool_name: str) -> list[ToolCall]:
        """Return all calls to a specific tool, preserving order."""
        return [call for call in self.calls if call.tool_name == tool_name]

    def call_count(self, tool_name: str) -> int:
        """Count how many times a specific tool was called."""
        return sum(1 for call in self.calls if call.tool_name == tool_name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        return {
            "calls": [call.to_dict() for call in self.calls],
            "model": self.model,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallTrajectory:
        """Reconstruct a trajectory from a serialized dict."""
        calls = []
        for call_data in data.get("calls", []):
            timestamp = None
            if "timestamp" in call_data:
                timestamp = datetime.fromisoformat(call_data["timestamp"])
            calls.append(ToolCall(
                tool_name=call_data["tool_name"],
                arguments=call_data.get("arguments", {}),
                call_index=call_data.get("call_index", 0),
                model=call_data.get("model", ""),
                provider=call_data.get("provider", ""),
                timestamp=timestamp,
                call_id=call_data.get("call_id"),
            ))
        return cls(
            calls=calls,
            model=data.get("model", ""),
            provider=data.get("provider", ""),
        )

    def __len__(self) -> int:
        return len(self.calls)
