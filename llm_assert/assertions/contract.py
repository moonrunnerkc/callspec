"""Contract assertions: per-tool argument validation.

Contract assertions verify that the arguments passed to specific tool calls
satisfy declared constraints. They operate on a ToolCallTrajectory and
target calls to a specific tool by name. Each assertion is applied to
every call to the target tool in the trajectory.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import jsonschema

from llm_assert.assertions.trajectory_base import TrajectoryAssertion
from llm_assert.core.config import LLMAssertConfig
from llm_assert.core.trajectory import ToolCall, ToolCallTrajectory
from llm_assert.core.types import IndividualAssertionResult


class ArgumentMatchesSchema(TrajectoryAssertion):
    """Passes if every call to the named tool has arguments matching a JSON Schema."""

    assertion_type = "contract"
    assertion_name = "argument_matches_schema"

    def __init__(self, tool_name: str, schema: dict[str, Any]) -> None:
        self._tool_name = tool_name
        self._schema = schema

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        calls = trajectory.calls_to(self._tool_name)
        if not calls:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot validate schema: tool '{self._tool_name}' "
                    f"not found in trajectory. Tools called: {trajectory.tool_names}."
                ),
                details={"tool_name": self._tool_name, "error": "tool_not_found"},
            )

        validator = jsonschema.Draft7Validator(self._schema)
        violations: list[dict[str, Any]] = []

        for call in calls:
            errors = list(validator.iter_errors(call.arguments))
            if errors:
                violations.append({
                    "call_index": call.call_index,
                    "arguments": call.arguments,
                    "errors": [
                        {
                            "path": list(err.absolute_path),
                            "message": err.message,
                            "validator": err.validator,
                        }
                        for err in errors
                    ],
                })

        if not violations:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"All {len(calls)} call(s) to '{self._tool_name}' "
                    f"match the declared schema."
                ),
            )

        first_violation = violations[0]
        first_error = first_violation["errors"][0]["message"]
        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"{len(violations)} of {len(calls)} call(s) to "
                f"'{self._tool_name}' violate the schema. "
                f"First violation at call_index {first_violation['call_index']}: "
                f"{first_error}."
            ),
            details={"violations": violations},
        )


class ArgumentContainsKey(TrajectoryAssertion):
    """Passes if every call to the named tool includes the specified argument key."""

    assertion_type = "contract"
    assertion_name = "argument_contains_key"

    def __init__(self, tool_name: str, key: str) -> None:
        self._tool_name = tool_name
        self._key = key

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        calls = trajectory.calls_to(self._tool_name)
        if not calls:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot check key '{self._key}': tool '{self._tool_name}' "
                    f"not found in trajectory."
                ),
                details={"tool_name": self._tool_name, "error": "tool_not_found"},
            )

        missing_in: list[int] = []
        for call in calls:
            if self._key not in call.arguments:
                missing_in.append(call.call_index)

        if not missing_in:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Key '{self._key}' present in all {len(calls)} "
                    f"call(s) to '{self._tool_name}'."
                ),
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Key '{self._key}' missing in {len(missing_in)} of "
                f"{len(calls)} call(s) to '{self._tool_name}'. "
                f"Missing at call indices: {missing_in}."
            ),
            details={
                "tool_name": self._tool_name,
                "key": self._key,
                "missing_at_indices": missing_in,
            },
        )


class ArgumentValueIn(TrajectoryAssertion):
    """Passes if the value for a key is in the allowed set, for all calls to the tool."""

    assertion_type = "contract"
    assertion_name = "argument_value_in"

    def __init__(self, tool_name: str, key: str, allowed_values: Sequence[Any]) -> None:
        self._tool_name = tool_name
        self._key = key
        self._allowed_values = list(allowed_values)

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        calls = trajectory.calls_to(self._tool_name)
        if not calls:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot check value for key '{self._key}': "
                    f"tool '{self._tool_name}' not found in trajectory."
                ),
                details={"tool_name": self._tool_name, "error": "tool_not_found"},
            )

        violations: list[dict[str, Any]] = []
        for call in calls:
            value = call.arguments.get(self._key)
            if value not in self._allowed_values:
                violations.append({
                    "call_index": call.call_index,
                    "actual_value": value,
                })

        if not violations:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Key '{self._key}' has allowed value in all "
                    f"{len(calls)} call(s) to '{self._tool_name}'."
                ),
            )

        first = violations[0]
        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Key '{self._key}' has disallowed value in {len(violations)} "
                f"of {len(calls)} call(s) to '{self._tool_name}'. "
                f"First violation at call_index {first['call_index']}: "
                f"got {first['actual_value']!r}, allowed: {self._allowed_values}."
            ),
            details={
                "violations": violations,
                "allowed_values": self._allowed_values,
            },
        )


class ArgumentMatchesPattern(TrajectoryAssertion):
    """Passes if the string value for a key matches a regex, for all calls to the tool."""

    assertion_type = "contract"
    assertion_name = "argument_matches_pattern"

    def __init__(self, tool_name: str, key: str, pattern: str) -> None:
        self._tool_name = tool_name
        self._key = key
        self._pattern = pattern
        self._compiled = re.compile(pattern)

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        calls = trajectory.calls_to(self._tool_name)
        if not calls:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot check pattern for key '{self._key}': "
                    f"tool '{self._tool_name}' not found in trajectory."
                ),
                details={"tool_name": self._tool_name, "error": "tool_not_found"},
            )

        violations: list[dict[str, Any]] = []
        for call in calls:
            if self._key not in call.arguments:
                violations.append({
                    "call_index": call.call_index,
                    "actual_value": None,
                    "reason": "key_absent",
                })
                continue

            value = call.arguments[self._key]
            if not isinstance(value, str):
                violations.append({
                    "call_index": call.call_index,
                    "actual_value": value,
                    "reason": "not_a_string",
                })
                continue

            if not self._compiled.search(value):
                violations.append({
                    "call_index": call.call_index,
                    "actual_value": value,
                    "reason": "no_match",
                })

        if not violations:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Key '{self._key}' matches pattern '{self._pattern}' "
                    f"in all {len(calls)} call(s) to '{self._tool_name}'."
                ),
            )

        first = violations[0]
        reason = first["reason"]
        if reason == "key_absent":
            detail = f"key '{self._key}' absent"
        elif reason == "not_a_string":
            detail = f"value is not a string (got {type(first['actual_value']).__name__})"
        else:
            detail = f"got {first['actual_value']!r}"

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Key '{self._key}' does not match pattern '{self._pattern}' "
                f"in {len(violations)} of {len(calls)} call(s) to '{self._tool_name}'. "
                f"First violation at call_index {first['call_index']}: {detail}."
            ),
            details={
                "violations": violations,
                "pattern": self._pattern,
            },
        )


class ArgumentNotEmpty(TrajectoryAssertion):
    """Passes if the value for a key is not empty/null/blank, for all calls to the tool."""

    assertion_type = "contract"
    assertion_name = "argument_not_empty"

    def __init__(self, tool_name: str, key: str) -> None:
        self._tool_name = tool_name
        self._key = key

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        calls = trajectory.calls_to(self._tool_name)
        if not calls:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot check emptiness for key '{self._key}': "
                    f"tool '{self._tool_name}' not found in trajectory."
                ),
                details={"tool_name": self._tool_name, "error": "tool_not_found"},
            )

        empty_at: list[int] = []
        for call in calls:
            value = call.arguments.get(self._key)
            if _is_empty(value):
                empty_at.append(call.call_index)

        if not empty_at:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Key '{self._key}' is non-empty in all "
                    f"{len(calls)} call(s) to '{self._tool_name}'."
                ),
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Key '{self._key}' is empty in {len(empty_at)} of "
                f"{len(calls)} call(s) to '{self._tool_name}'. "
                f"Empty at call indices: {empty_at}."
            ),
            details={
                "tool_name": self._tool_name,
                "key": self._key,
                "empty_at_indices": empty_at,
            },
        )


class CustomContract(TrajectoryAssertion):
    """Passes if a user-supplied predicate returns True for every call to the tool."""

    assertion_type = "contract"
    assertion_name = "custom_contract"

    def __init__(
        self,
        tool_name: str,
        predicate_fn: Callable[[ToolCall], bool],
        description: str = "custom validation",
    ) -> None:
        self._tool_name = tool_name
        self._predicate_fn = predicate_fn
        self._description = description

    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        calls = trajectory.calls_to(self._tool_name)
        if not calls:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot apply custom contract ({self._description}): "
                    f"tool '{self._tool_name}' not found in trajectory."
                ),
                details={"tool_name": self._tool_name, "error": "tool_not_found"},
            )

        failed_at: list[int] = []
        for call in calls:
            if not self._predicate_fn(call):
                failed_at.append(call.call_index)

        if not failed_at:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Custom contract ({self._description}) passed for all "
                    f"{len(calls)} call(s) to '{self._tool_name}'."
                ),
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Custom contract ({self._description}) failed for "
                f"{len(failed_at)} of {len(calls)} call(s) to "
                f"'{self._tool_name}'. Failed at call indices: {failed_at}."
            ),
            details={
                "tool_name": self._tool_name,
                "description": self._description,
                "failed_at_indices": failed_at,
            },
        )


def _is_empty(value: Any) -> bool:
    """Check if a value is empty, null, or blank whitespace."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False
