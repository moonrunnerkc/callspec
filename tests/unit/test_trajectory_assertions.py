"""Tests for trajectory assertions (sequence-level tool call validation)."""

import pytest

from llm_assert.assertions.trajectory import (
    CallCount,
    CallsExactly,
    CallsSubset,
    CallsTool,
    CallsToolsInOrder,
    DoesNotCall,
    NoRepeatedCalls,
)
from llm_assert.core.config import LLMAssertConfig
from llm_assert.core.trajectory import ToolCall, ToolCallTrajectory


# -- Helpers --

def _trajectory(*tool_names: str) -> ToolCallTrajectory:
    """Build a trajectory from tool name strings."""
    calls = [
        ToolCall(tool_name=name, arguments={}, call_index=i)
        for i, name in enumerate(tool_names)
    ]
    return ToolCallTrajectory(calls=calls, model="test", provider="mock")


EMPTY = ToolCallTrajectory(calls=[], model="test", provider="mock")
CONFIG = LLMAssertConfig()


# ── CallsTool ──

class TestCallsTool:
    def test_passes_when_tool_present(self):
        result = CallsTool("search").evaluate_trajectory(_trajectory("search", "book"), CONFIG)
        assert result.passed
        assert result.assertion_name == "calls_tool"
        assert result.assertion_type == "trajectory"

    def test_passes_when_tool_appears_multiple_times(self):
        result = CallsTool("search").evaluate_trajectory(_trajectory("search", "search"), CONFIG)
        assert result.passed

    def test_fails_when_tool_absent(self):
        result = CallsTool("delete").evaluate_trajectory(_trajectory("search", "book"), CONFIG)
        assert not result.passed
        assert "delete" in result.message

    def test_fails_on_empty_trajectory(self):
        result = CallsTool("search").evaluate_trajectory(EMPTY, CONFIG)
        assert not result.passed

    def test_exact_name_match_required(self):
        # "search_v2" should not match "search"
        result = CallsTool("search").evaluate_trajectory(_trajectory("search_v2"), CONFIG)
        assert not result.passed


# ── CallsToolsInOrder ──

class TestCallsToolsInOrder:
    def test_passes_exact_match(self):
        result = CallsToolsInOrder(["search", "book"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert result.passed

    def test_passes_with_interleaved_tools(self):
        result = CallsToolsInOrder(["search", "book"]).evaluate_trajectory(
            _trajectory("search", "validate", "book"), CONFIG
        )
        assert result.passed

    def test_fails_wrong_order(self):
        result = CallsToolsInOrder(["book", "search"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert not result.passed

    def test_fails_when_tool_missing(self):
        result = CallsToolsInOrder(["search", "delete"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert not result.passed
        assert "delete" in result.message

    def test_passes_single_tool(self):
        result = CallsToolsInOrder(["search"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert result.passed

    def test_fails_on_empty_trajectory(self):
        result = CallsToolsInOrder(["search"]).evaluate_trajectory(EMPTY, CONFIG)
        assert not result.passed

    def test_three_tools_in_order(self):
        result = CallsToolsInOrder(["a", "b", "c"]).evaluate_trajectory(
            _trajectory("a", "x", "b", "y", "c"), CONFIG
        )
        assert result.passed


# ── CallsExactly ──

class TestCallsExactly:
    def test_passes_exact_sequence(self):
        result = CallsExactly(["search", "book"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert result.passed

    def test_fails_extra_tool(self):
        result = CallsExactly(["search", "book"]).evaluate_trajectory(
            _trajectory("search", "validate", "book"), CONFIG
        )
        assert not result.passed

    def test_fails_missing_tool(self):
        result = CallsExactly(["search", "book"]).evaluate_trajectory(
            _trajectory("search"), CONFIG
        )
        assert not result.passed

    def test_fails_wrong_order(self):
        result = CallsExactly(["book", "search"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert not result.passed

    def test_passes_empty_expected_empty_actual(self):
        result = CallsExactly([]).evaluate_trajectory(EMPTY, CONFIG)
        assert result.passed

    def test_fails_empty_expected_nonempty_actual(self):
        result = CallsExactly([]).evaluate_trajectory(_trajectory("search"), CONFIG)
        assert not result.passed


# ── CallsSubset ──

class TestCallsSubset:
    def test_passes_all_present(self):
        result = CallsSubset(["search", "book"]).evaluate_trajectory(
            _trajectory("search", "validate", "book"), CONFIG
        )
        assert result.passed

    def test_passes_order_irrelevant(self):
        result = CallsSubset(["book", "search"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert result.passed

    def test_fails_when_one_missing(self):
        result = CallsSubset(["search", "delete"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert not result.passed
        assert "delete" in result.message

    def test_passes_single_required(self):
        result = CallsSubset(["search"]).evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert result.passed

    def test_fails_on_empty_trajectory(self):
        result = CallsSubset(["search"]).evaluate_trajectory(EMPTY, CONFIG)
        assert not result.passed

    def test_passes_empty_required(self):
        # Empty required set is trivially satisfied
        result = CallsSubset([]).evaluate_trajectory(_trajectory("search"), CONFIG)
        assert result.passed


# ── DoesNotCall ──

class TestDoesNotCall:
    def test_passes_when_absent(self):
        result = DoesNotCall("delete").evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert result.passed

    def test_fails_when_present(self):
        result = DoesNotCall("search").evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert not result.passed

    def test_passes_on_empty_trajectory(self):
        result = DoesNotCall("search").evaluate_trajectory(EMPTY, CONFIG)
        assert result.passed


# ── CallCount ──

class TestCallCount:
    def test_exact_count(self):
        result = CallCount("search", min_count=2, max_count=2).evaluate_trajectory(
            _trajectory("search", "search"), CONFIG
        )
        assert result.passed

    def test_min_only(self):
        result = CallCount("search", min_count=1).evaluate_trajectory(
            _trajectory("search", "search", "search"), CONFIG
        )
        assert result.passed

    def test_max_only(self):
        result = CallCount("search", min_count=0, max_count=2).evaluate_trajectory(
            _trajectory("search"), CONFIG
        )
        assert result.passed

    def test_fails_below_min(self):
        result = CallCount("search", min_count=3).evaluate_trajectory(
            _trajectory("search", "search"), CONFIG
        )
        assert not result.passed

    def test_fails_above_max(self):
        result = CallCount("search", min_count=0, max_count=1).evaluate_trajectory(
            _trajectory("search", "search"), CONFIG
        )
        assert not result.passed

    def test_tool_not_called_passes_when_min_zero(self):
        result = CallCount("delete", min_count=0, max_count=3).evaluate_trajectory(
            _trajectory("search"), CONFIG
        )
        assert result.passed

    def test_tool_not_called_fails_when_min_above_zero(self):
        result = CallCount("delete", min_count=1).evaluate_trajectory(
            _trajectory("search"), CONFIG
        )
        assert not result.passed

    def test_message_includes_count(self):
        result = CallCount("search", min_count=5).evaluate_trajectory(
            _trajectory("search", "search"), CONFIG
        )
        assert "2" in result.message


# ── NoRepeatedCalls ──

class TestNoRepeatedCalls:
    def test_passes_single_call(self):
        result = NoRepeatedCalls("search").evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert result.passed

    def test_fails_multiple_calls(self):
        result = NoRepeatedCalls("search").evaluate_trajectory(
            _trajectory("search", "book", "search"), CONFIG
        )
        assert not result.passed

    def test_passes_tool_not_called(self):
        result = NoRepeatedCalls("delete").evaluate_trajectory(
            _trajectory("search", "book"), CONFIG
        )
        assert result.passed

    def test_passes_on_empty_trajectory(self):
        result = NoRepeatedCalls("search").evaluate_trajectory(EMPTY, CONFIG)
        assert result.passed
