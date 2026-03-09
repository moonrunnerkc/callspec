"""TrajectoryBuilder: fluent API for chaining trajectory and contract assertions.

Works like AssertionBuilder but operates on a ToolCallTrajectory instead of
content strings. Users get a builder from verdict.assert_trajectory(trajectory)
and chain assertions until .run().
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from callspec.assertions.contract import (
    ArgumentContainsKey,
    ArgumentMatchesPattern,
    ArgumentMatchesSchema,
    ArgumentNotEmpty,
    ArgumentValueIn,
    CustomContract,
)
from callspec.assertions.trajectory import (
    CallCount,
    CallsExactly,
    CallsSubset,
    CallsTool,
    CallsToolsInOrder,
    DoesNotCall,
    NoRepeatedCalls,
)
from callspec.assertions.trajectory_base import TrajectoryAssertion
from callspec.assertions.trajectory_regression import (
    MatchesTrajectoryBaseline,
    TrajectorySequenceMatches,
)
from callspec.core.config import CallspecConfig
from callspec.core.trajectory import ToolCall, ToolCallTrajectory
from callspec.core.types import (
    AssertionResult,
    IndividualAssertionResult,
    ProviderResponse,
)
from callspec.snapshots.manager import SnapshotManager


class TrajectoryBuilder:
    """Fluent builder for trajectory and contract assertion chains.

    Each method appends an assertion and returns self. Call .run() to
    evaluate all assertions against the trajectory and get a result.

    Usage:
        result = (
            TrajectoryBuilder(trajectory)
            .calls_tools_in_order(["search", "book"])
            .argument_not_empty("search", "query")
            .does_not_call("delete")
            .run()
        )
    """

    def __init__(
        self,
        trajectory: ToolCallTrajectory,
        config: CallspecConfig | None = None,
    ) -> None:
        self._trajectory = trajectory
        self._config = config or CallspecConfig()
        self._assertions: list[TrajectoryAssertion] = []

    # -- Trajectory assertions --

    def calls_tool(self, tool_name: str) -> TrajectoryBuilder:
        """Assert the trajectory contains at least one call to this tool."""
        self._assertions.append(CallsTool(tool_name))
        return self

    def calls_tools_in_order(self, expected_order: Sequence[str]) -> TrajectoryBuilder:
        """Assert the trajectory calls these tools in this relative order."""
        self._assertions.append(CallsToolsInOrder(expected_order))
        return self

    def calls_exactly(self, expected_tools: Sequence[str]) -> TrajectoryBuilder:
        """Assert the trajectory calls exactly these tools in this exact order."""
        self._assertions.append(CallsExactly(expected_tools))
        return self

    def calls_subset(self, required_tools: Sequence[str]) -> TrajectoryBuilder:
        """Assert every listed tool appears at least once (order doesn't matter)."""
        self._assertions.append(CallsSubset(required_tools))
        return self

    def does_not_call(self, tool_name: str) -> TrajectoryBuilder:
        """Assert the trajectory never calls this tool."""
        self._assertions.append(DoesNotCall(tool_name))
        return self

    def call_count(
        self,
        tool_name: str,
        min_count: int = 0,
        max_count: int | None = None,
    ) -> TrajectoryBuilder:
        """Assert the tool is called between min_count and max_count times."""
        self._assertions.append(CallCount(tool_name, min_count, max_count))
        return self

    def no_repeated_calls(self, tool_name: str) -> TrajectoryBuilder:
        """Assert the tool is called at most once."""
        self._assertions.append(NoRepeatedCalls(tool_name))
        return self

    # -- Contract assertions --

    def argument_matches_schema(
        self, tool_name: str, schema: dict[str, Any]
    ) -> TrajectoryBuilder:
        """Assert every call to this tool has arguments matching the JSON Schema."""
        self._assertions.append(ArgumentMatchesSchema(tool_name, schema))
        return self

    def argument_contains_key(self, tool_name: str, key: str) -> TrajectoryBuilder:
        """Assert every call to this tool includes this argument key."""
        self._assertions.append(ArgumentContainsKey(tool_name, key))
        return self

    def argument_value_in(
        self, tool_name: str, key: str, allowed_values: Sequence[Any]
    ) -> TrajectoryBuilder:
        """Assert the value for this key is in the allowed set."""
        self._assertions.append(ArgumentValueIn(tool_name, key, allowed_values))
        return self

    def argument_matches_pattern(
        self, tool_name: str, key: str, pattern: str
    ) -> TrajectoryBuilder:
        """Assert the string value for this key matches the regex pattern."""
        self._assertions.append(ArgumentMatchesPattern(tool_name, key, pattern))
        return self

    def argument_not_empty(self, tool_name: str, key: str) -> TrajectoryBuilder:
        """Assert the value for this key is not empty/null/blank."""
        self._assertions.append(ArgumentNotEmpty(tool_name, key))
        return self

    def custom_contract(
        self,
        tool_name: str,
        predicate_fn: Callable[[ToolCall], bool],
        description: str = "custom validation",
    ) -> TrajectoryBuilder:
        """Assert a user-supplied predicate passes for every call to the tool."""
        self._assertions.append(CustomContract(tool_name, predicate_fn, description))
        return self

    # -- Regression assertions --

    def matches_baseline(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
    ) -> TrajectoryBuilder:
        """Assert the trajectory matches a recorded baseline (sequence + argument keys)."""
        self._assertions.append(
            MatchesTrajectoryBaseline(snapshot_key, snapshot_manager)
        )
        return self

    def sequence_matches_baseline(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
    ) -> TrajectoryBuilder:
        """Assert the tool name sequence matches the baseline (arguments ignored)."""
        self._assertions.append(
            TrajectorySequenceMatches(snapshot_key, snapshot_manager)
        )
        return self

    # -- Execution --

    def run(self) -> AssertionResult:
        """Evaluate all chained assertions against the trajectory.

        Returns an AssertionResult. The provider_response is a synthetic
        ProviderResponse built from the trajectory metadata since trajectory
        assertions do not invoke a provider directly.
        """
        start_time = time.monotonic()

        individual_results: list[IndividualAssertionResult] = []
        all_passed = True

        for assertion in self._assertions:
            individual = assertion.evaluate_trajectory(self._trajectory, self._config)
            individual_results.append(individual)

            if not individual.passed:
                all_passed = False
                if self._config.fail_fast:
                    break

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Build a synthetic ProviderResponse from trajectory metadata
        # so the result shape is consistent with content-based assertions
        synthetic_response = ProviderResponse(
            content="",
            raw=self._trajectory.raw_response,
            model=self._trajectory.model,
            provider=self._trajectory.provider,
            tool_calls=[c.to_dict() for c in self._trajectory.calls],
        )

        return AssertionResult(
            passed=all_passed,
            assertions=individual_results,
            provider_response=synthetic_response,
            execution_time_ms=elapsed_ms,
            model=self._trajectory.model,
            prompt_tokens=None,
            completion_tokens=None,
        )

    @property
    def assertion_count(self) -> int:
        """Number of assertions in the current chain."""
        return len(self._assertions)
