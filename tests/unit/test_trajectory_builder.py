"""Tests for TrajectoryBuilder fluent API and Callspec.assert_trajectory integration."""

import pytest

from callspec import Callspec, ToolCall, ToolCallTrajectory, TrajectoryBuilder
from callspec.core.config import CallspecConfig
from callspec.providers.mock import MockProvider


# -- Helpers --

def _trajectory(*calls_spec: tuple[str, dict]) -> ToolCallTrajectory:
    """Build trajectory from (tool_name, arguments) pairs."""
    calls = [
        ToolCall(tool_name=name, arguments=args, call_index=i)
        for i, (name, args) in enumerate(calls_spec)
    ]
    return ToolCallTrajectory(calls=calls, model="gpt-4o", provider="openai")


SEARCH_BOOK = _trajectory(
    ("search", {"query": "python testing", "limit": 10}),
    ("book", {"id": 42, "date": "2025-01-15"}),
)

SEARCH_SCHEMA = {
    "type": "object",
    "required": ["query"],
    "properties": {
        "query": {"type": "string"},
        "limit": {"type": "integer"},
    },
}


# ── TrajectoryBuilder basics ──

class TestTrajectoryBuilderBasics:
    def test_empty_chain_passes(self):
        result = TrajectoryBuilder(SEARCH_BOOK).run()
        assert result.passed
        assert len(result.assertions) == 0

    def test_assertion_count(self):
        builder = (
            TrajectoryBuilder(SEARCH_BOOK)
            .calls_tool("search")
            .does_not_call("delete")
        )
        assert builder.assertion_count == 2

    def test_result_has_model_and_provider(self):
        result = TrajectoryBuilder(SEARCH_BOOK).calls_tool("search").run()
        assert result.model == "gpt-4o"
        assert result.provider_response.provider == "openai"

    def test_result_has_tool_calls_in_response(self):
        result = TrajectoryBuilder(SEARCH_BOOK).calls_tool("search").run()
        assert len(result.provider_response.tool_calls) == 2
        assert result.provider_response.tool_calls[0]["tool_name"] == "search"

    def test_execution_time_is_nonnegative(self):
        result = TrajectoryBuilder(SEARCH_BOOK).calls_tool("search").run()
        assert result.execution_time_ms >= 0

    def test_tokens_are_none(self):
        # Trajectory assertions dont involve provider calls, so no token counts
        result = TrajectoryBuilder(SEARCH_BOOK).calls_tool("search").run()
        assert result.prompt_tokens is None
        assert result.completion_tokens is None


# ── Chaining trajectory assertions ──

class TestTrajectoryBuilderTrajectoryChains:
    def test_multiple_passing_assertions(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .calls_tool("search")
            .calls_tool("book")
            .calls_tools_in_order(["search", "book"])
            .does_not_call("delete")
            .run()
        )
        assert result.passed
        assert len(result.assertions) == 4
        assert all(a.passed for a in result.assertions)

    def test_one_failure_fails_result(self):
        config = CallspecConfig(fail_fast=False)
        result = (
            TrajectoryBuilder(SEARCH_BOOK, config=config)
            .calls_tool("search")
            .calls_tool("nonexistent")
            .calls_tool("book")
            .run()
        )
        assert not result.passed
        assert result.assertions[0].passed  # search
        assert not result.assertions[1].passed  # nonexistent
        assert result.assertions[2].passed  # book

    def test_fail_fast_stops_at_first_failure(self):
        config = CallspecConfig(fail_fast=True)
        result = (
            TrajectoryBuilder(SEARCH_BOOK, config=config)
            .calls_tool("nonexistent")
            .calls_tool("search")
            .run()
        )
        assert not result.passed
        # second assertion not evaluated due to fail_fast
        assert len(result.assertions) == 1

    def test_calls_exactly_pass(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .calls_exactly(["search", "book"])
            .run()
        )
        assert result.passed

    def test_calls_subset_pass(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .calls_subset(["book"])
            .run()
        )
        assert result.passed

    def test_call_count_pass(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .call_count("search", min_count=1, max_count=1)
            .run()
        )
        assert result.passed

    def test_no_repeated_calls_pass(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .no_repeated_calls("search")
            .run()
        )
        assert result.passed


# ── Chaining contract assertions ──

class TestTrajectoryBuilderContractChains:
    def test_schema_validation(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .argument_matches_schema("search", SEARCH_SCHEMA)
            .run()
        )
        assert result.passed

    def test_contains_key(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .argument_contains_key("search", "query")
            .argument_contains_key("book", "id")
            .run()
        )
        assert result.passed

    def test_value_in(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .argument_value_in("search", "limit", [5, 10, 20])
            .run()
        )
        assert result.passed

    def test_value_in_fails(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .argument_value_in("search", "limit", [5, 20])
            .run()
        )
        assert not result.passed

    def test_matches_pattern(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .argument_matches_pattern("search", "query", r"python")
            .run()
        )
        assert result.passed

    def test_not_empty(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .argument_not_empty("search", "query")
            .run()
        )
        assert result.passed

    def test_custom_contract(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .custom_contract(
                "search",
                lambda call: call.arguments.get("limit", 0) <= 100,
                "limit must be at most 100",
            )
            .run()
        )
        assert result.passed


# ── Mixed trajectory + contract chains ──

class TestMixedChains:
    def test_full_chain_passes(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .calls_tools_in_order(["search", "book"])
            .argument_matches_schema("search", SEARCH_SCHEMA)
            .argument_not_empty("search", "query")
            .does_not_call("delete")
            .call_count("search", min_count=1, max_count=3)
            .run()
        )
        assert result.passed
        assert len(result.assertions) == 5

    def test_trajectory_pass_contract_fail(self):
        bad_schema = {
            "type": "object",
            "required": ["nonexistent_field"],
        }
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .calls_tool("search")
            .argument_matches_schema("search", bad_schema)
            .run()
        )
        assert not result.passed
        assert result.assertions[0].passed  # calls_tool
        assert not result.assertions[1].passed  # schema

    def test_assertion_types_propagated(self):
        result = (
            TrajectoryBuilder(SEARCH_BOOK)
            .calls_tool("search")
            .argument_not_empty("search", "query")
            .run()
        )
        assert result.assertions[0].assertion_type == "trajectory"
        assert result.assertions[1].assertion_type == "contract"


# ── Callspec.assert_trajectory integration ──

class TestCallspecIntegration:
    def test_assert_trajectory_returns_builder(self):
        provider = MockProvider(response_fn=lambda p, m: "ok")
        v = Callspec(provider)
        builder = v.assert_trajectory(SEARCH_BOOK)
        assert isinstance(builder, TrajectoryBuilder)

    def test_inherits_config(self):
        config = CallspecConfig(fail_fast=True)
        provider = MockProvider(response_fn=lambda p, m: "ok")
        v = Callspec(provider, config=config)
        builder = v.assert_trajectory(SEARCH_BOOK)
        # fail_fast should propagate: first failure stops evaluation
        result = (
            builder
            .calls_tool("nonexistent")
            .calls_tool("search")
            .run()
        )
        assert not result.passed
        assert len(result.assertions) == 1

    def test_full_chain_via_llmassert(self):
        provider = MockProvider(response_fn=lambda p, m: "ok")
        v = Callspec(provider)
        result = (
            v.assert_trajectory(SEARCH_BOOK)
            .calls_tools_in_order(["search", "book"])
            .argument_matches_schema("search", SEARCH_SCHEMA)
            .does_not_call("delete")
            .run()
        )
        assert result.passed
        assert result.model == "gpt-4o"


# ── Empty trajectory edge cases ──

class TestEmptyTrajectory:
    EMPTY = ToolCallTrajectory(calls=[], model="test", provider="mock")

    def test_calls_tool_fails(self):
        result = TrajectoryBuilder(self.EMPTY).calls_tool("search").run()
        assert not result.passed

    def test_does_not_call_passes(self):
        result = TrajectoryBuilder(self.EMPTY).does_not_call("search").run()
        assert result.passed

    def test_calls_exactly_empty_passes(self):
        result = TrajectoryBuilder(self.EMPTY).calls_exactly([]).run()
        assert result.passed

    def test_contract_fails_on_empty(self):
        result = (
            TrajectoryBuilder(self.EMPTY)
            .argument_not_empty("search", "query")
            .run()
        )
        assert not result.passed
