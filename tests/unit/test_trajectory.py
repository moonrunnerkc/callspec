"""Tests for ToolCall and ToolCallTrajectory data models.

Covers construction, serialization (to_dict / from_dict), helper methods,
edge cases with empty trajectories, and frozen dataclass immutability.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from callspec.core.trajectory import ToolCall, ToolCallTrajectory

# ---------------------------------------------------------------------------
# ToolCall construction and serialization
# ---------------------------------------------------------------------------

class TestToolCall:
    """ToolCall immutable record and serialization."""

    def test_minimal_construction(self):
        call = ToolCall(tool_name="search")
        assert call.tool_name == "search"
        assert call.arguments == {}
        assert call.call_index == 0
        assert call.model == ""
        assert call.provider == ""
        assert call.timestamp is None
        assert call.call_id is None

    def test_full_construction(self):
        ts = datetime(2026, 3, 15, 10, 30, 0)
        call = ToolCall(
            tool_name="book_flight",
            arguments={"flight_id": "BA123", "passenger": "Brad"},
            call_index=2,
            model="gpt-4o-2024-11-20",
            provider="openai",
            timestamp=ts,
            raw={"original": "data"},
            call_id="call_xyz789",
        )
        assert call.tool_name == "book_flight"
        assert call.arguments == {"flight_id": "BA123", "passenger": "Brad"}
        assert call.call_index == 2
        assert call.model == "gpt-4o-2024-11-20"
        assert call.provider == "openai"
        assert call.timestamp == ts
        assert call.raw == {"original": "data"}
        assert call.call_id == "call_xyz789"

    def test_frozen_immutability(self):
        call = ToolCall(tool_name="search")
        with pytest.raises(AttributeError):
            call.tool_name = "modified"  # type: ignore[misc]

    def test_to_dict_minimal(self):
        call = ToolCall(tool_name="search")
        expected = {
            "tool_name": "search",
            "arguments": {},
            "call_index": 0,
        }
        assert call.to_dict() == expected

    def test_to_dict_includes_optional_fields_when_set(self):
        ts = datetime(2026, 3, 15, 10, 30, 0)
        call = ToolCall(
            tool_name="book",
            arguments={"id": 1},
            call_index=1,
            model="gpt-4o",
            provider="openai",
            timestamp=ts,
            call_id="call_abc",
        )
        result = call.to_dict()
        assert result["model"] == "gpt-4o"
        assert result["provider"] == "openai"
        assert result["timestamp"] == "2026-03-15T10:30:00"
        assert result["call_id"] == "call_abc"

    def test_to_dict_omits_empty_optional_fields(self):
        call = ToolCall(tool_name="search", arguments={"q": "test"})
        result = call.to_dict()
        assert "model" not in result
        assert "provider" not in result
        assert "timestamp" not in result
        assert "call_id" not in result

    def test_to_dict_omits_raw_field(self):
        """raw is for debugging only, not serialized."""
        call = ToolCall(tool_name="x", raw={"big": "payload"})
        assert "raw" not in call.to_dict()


# ---------------------------------------------------------------------------
# ToolCallTrajectory
# ---------------------------------------------------------------------------

class TestToolCallTrajectory:
    """Trajectory construction, helpers, and serialization round-trip."""

    @pytest.fixture()
    def three_call_trajectory(self) -> ToolCallTrajectory:
        """Trajectory with three calls to two distinct tools."""
        return ToolCallTrajectory(
            calls=[
                ToolCall(tool_name="search", arguments={"q": "flights"}, call_index=0),
                ToolCall(tool_name="book", arguments={"id": "BA123"}, call_index=1),
                ToolCall(tool_name="search", arguments={"q": "hotels"}, call_index=2),
            ],
            model="gpt-4o",
            provider="openai",
        )

    def test_empty_trajectory(self):
        traj = ToolCallTrajectory()
        assert traj.is_empty
        assert len(traj) == 0
        assert traj.tool_names == []

    def test_tool_names_property(self, three_call_trajectory: ToolCallTrajectory):
        assert three_call_trajectory.tool_names == ["search", "book", "search"]

    def test_is_empty_false_when_populated(self, three_call_trajectory: ToolCallTrajectory):
        assert not three_call_trajectory.is_empty

    def test_len(self, three_call_trajectory: ToolCallTrajectory):
        assert len(three_call_trajectory) == 3

    def test_calls_to_returns_matching(self, three_call_trajectory: ToolCallTrajectory):
        search_calls = three_call_trajectory.calls_to("search")
        assert len(search_calls) == 2
        assert all(c.tool_name == "search" for c in search_calls)

    def test_calls_to_nonexistent_returns_empty(self, three_call_trajectory: ToolCallTrajectory):
        assert three_call_trajectory.calls_to("delete") == []

    def test_call_count(self, three_call_trajectory: ToolCallTrajectory):
        assert three_call_trajectory.call_count("search") == 2
        assert three_call_trajectory.call_count("book") == 1
        assert three_call_trajectory.call_count("nonexistent") == 0

    def test_to_dict(self, three_call_trajectory: ToolCallTrajectory):
        data = three_call_trajectory.to_dict()
        assert data["model"] == "gpt-4o"
        assert data["provider"] == "openai"
        assert len(data["calls"]) == 3
        assert data["calls"][0]["tool_name"] == "search"
        assert data["calls"][1]["tool_name"] == "book"

    def test_from_dict_round_trip(self, three_call_trajectory: ToolCallTrajectory):
        serialized = three_call_trajectory.to_dict()
        restored = ToolCallTrajectory.from_dict(serialized)
        assert len(restored) == 3
        assert restored.tool_names == ["search", "book", "search"]
        assert restored.model == "gpt-4o"
        assert restored.provider == "openai"
        assert restored.calls[1].arguments == {"id": "BA123"}

    def test_from_dict_with_timestamps(self):
        data = {
            "calls": [
                {
                    "tool_name": "search",
                    "arguments": {},
                    "call_index": 0,
                    "timestamp": "2026-03-15T10:30:00",
                }
            ],
            "model": "test",
            "provider": "test",
        }
        traj = ToolCallTrajectory.from_dict(data)
        assert traj.calls[0].timestamp == datetime(2026, 3, 15, 10, 30, 0)

    def test_from_dict_empty_calls(self):
        data = {"calls": [], "model": "empty", "provider": "test"}
        traj = ToolCallTrajectory.from_dict(data)
        assert traj.is_empty
        assert traj.model == "empty"

    def test_from_dict_missing_keys_uses_defaults(self):
        data = {"calls": [{"tool_name": "x"}]}
        traj = ToolCallTrajectory.from_dict(data)
        assert len(traj) == 1
        assert traj.calls[0].arguments == {}
        assert traj.calls[0].call_index == 0
        assert traj.model == ""
