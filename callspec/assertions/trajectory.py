"""Trajectory assertions: verify tool-call sequence shape and ordering.

These assertions operate on a ToolCallTrajectory and check properties
of the overall sequence: which tools were called, in what order, how
many times. They do not inspect individual tool arguments (that is
contract.py's job).
"""

from __future__ import annotations

from typing import Sequence

from callspec.assertions.trajectory_base import TrajectoryAssertion
from callspec.core.config import CallspecConfig
from callspec.core.trajectory import ToolCallTrajectory
from callspec.core.types import IndividualAssertionResult


class CallsTool(TrajectoryAssertion):
    """Passes if the trajectory contains at least one call to the named tool."""

    assertion_name = "calls_tool"

    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: CallspecConfig,
    ) -> IndividualAssertionResult:
        found = trajectory.call_count(self._tool_name)
        if found > 0:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Tool '{self._tool_name}' called {found} time(s) "
                    f"in trajectory of {len(trajectory)} calls."
                ),
            )
        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Tool '{self._tool_name}' not found in trajectory. "
                f"Tools called: {trajectory.tool_names}."
            ),
            details={"expected_tool": self._tool_name, "actual_tools": trajectory.tool_names},
        )


class CallsToolsInOrder(TrajectoryAssertion):
    """Passes if the trajectory calls the listed tools in relative order.

    Other tools may appear between the expected tools. Only the relative
    ordering matters: [A, B, C] passes if A appears before B and B
    appears before C in the trajectory, regardless of what else is
    in between.
    """

    assertion_name = "calls_tools_in_order"

    def __init__(self, expected_order: Sequence[str]) -> None:
        self._expected_order = list(expected_order)

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: CallspecConfig,
    ) -> IndividualAssertionResult:
        actual_names = trajectory.tool_names
        search_from = 0
        matched: list[str] = []

        for expected_name in self._expected_order:
            found = False
            for i in range(search_from, len(actual_names)):
                if actual_names[i] == expected_name:
                    matched.append(expected_name)
                    search_from = i + 1
                    found = True
                    break
            if not found:
                return IndividualAssertionResult(
                    assertion_type=self.assertion_type,
                    assertion_name=self.assertion_name,
                    passed=False,
                    message=(
                        f"Expected tools in order {self._expected_order}. "
                        f"Matched {matched} but '{expected_name}' not found "
                        f"after position {search_from}. "
                        f"Actual trajectory: {actual_names}."
                    ),
                    details={
                        "expected_order": self._expected_order,
                        "actual_tools": actual_names,
                        "matched_prefix": matched,
                        "missing": expected_name,
                    },
                )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=True,
            message=(
                f"Tools called in expected order: {self._expected_order}. "
                f"Actual trajectory: {actual_names}."
            ),
        )


class CallsExactly(TrajectoryAssertion):
    """Passes if the trajectory calls exactly these tools in this exact order.

    No extra calls, no missing calls, exact positional match.
    """

    assertion_name = "calls_exactly"

    def __init__(self, expected_tools: Sequence[str]) -> None:
        self._expected_tools = list(expected_tools)

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: CallspecConfig,
    ) -> IndividualAssertionResult:
        actual = trajectory.tool_names
        if actual == self._expected_tools:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=f"Trajectory exactly matches: {self._expected_tools}.",
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Trajectory does not match expected sequence. "
                f"Expected: {self._expected_tools}. "
                f"Actual: {actual}."
            ),
            details={"expected": self._expected_tools, "actual": actual},
        )


class CallsSubset(TrajectoryAssertion):
    """Passes if every listed tool appears at least once. Order does not matter."""

    assertion_name = "calls_subset"

    def __init__(self, required_tools: Sequence[str]) -> None:
        self._required_tools = list(required_tools)

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: CallspecConfig,
    ) -> IndividualAssertionResult:
        actual_set = set(trajectory.tool_names)
        missing = [t for t in self._required_tools if t not in actual_set]

        if not missing:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"All required tools present: {self._required_tools}. "
                    f"Actual trajectory: {trajectory.tool_names}."
                ),
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Missing required tools: {missing}. "
                f"Required: {self._required_tools}. "
                f"Actual trajectory: {trajectory.tool_names}."
            ),
            details={
                "required": self._required_tools,
                "missing": missing,
                "actual_tools": trajectory.tool_names,
            },
        )


class DoesNotCall(TrajectoryAssertion):
    """Passes if the trajectory never calls the named tool."""

    assertion_name = "does_not_call"

    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: CallspecConfig,
    ) -> IndividualAssertionResult:
        count = trajectory.call_count(self._tool_name)
        if count == 0:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Tool '{self._tool_name}' not called. "
                    f"Trajectory: {trajectory.tool_names}."
                ),
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Tool '{self._tool_name}' was called {count} time(s) "
                f"but should not have been called. "
                f"Trajectory: {trajectory.tool_names}."
            ),
            details={
                "forbidden_tool": self._tool_name,
                "call_count": count,
                "actual_tools": trajectory.tool_names,
            },
        )


class CallCount(TrajectoryAssertion):
    """Passes if the tool is called between min_count and max_count times (inclusive)."""

    assertion_name = "call_count"

    def __init__(
        self,
        tool_name: str,
        min_count: int = 0,
        max_count: int | None = None,
    ) -> None:
        self._tool_name = tool_name
        self._min_count = min_count
        self._max_count = max_count

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: CallspecConfig,
    ) -> IndividualAssertionResult:
        actual = trajectory.call_count(self._tool_name)
        max_display = self._max_count if self._max_count is not None else "unbounded"

        in_range = actual >= self._min_count
        if self._max_count is not None:
            in_range = in_range and actual <= self._max_count

        if in_range:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Tool '{self._tool_name}' called {actual} time(s), "
                    f"within range [{self._min_count}, {max_display}]."
                ),
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Tool '{self._tool_name}' called {actual} time(s), "
                f"expected range [{self._min_count}, {max_display}]."
            ),
            details={
                "tool_name": self._tool_name,
                "actual_count": actual,
                "min_count": self._min_count,
                "max_count": self._max_count,
            },
        )


class NoRepeatedCalls(TrajectoryAssertion):
    """Passes if the named tool is called at most once."""

    assertion_name = "no_repeated_calls"

    def __init__(self, tool_name: str) -> None:
        self._tool_name = tool_name

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: CallspecConfig,
    ) -> IndividualAssertionResult:
        count = trajectory.call_count(self._tool_name)
        if count <= 1:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Tool '{self._tool_name}' called {count} time(s). "
                    f"No repeated calls."
                ),
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Tool '{self._tool_name}' called {count} times "
                f"but should be called at most once."
            ),
            details={
                "tool_name": self._tool_name,
                "actual_count": count,
            },
        )
