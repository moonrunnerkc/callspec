"""YAML suite parser: loads assertion suites from YAML configuration files.

Both the Python API and the YAML format compile to the same internal
representation (AssertionSuite). This module handles the YAML-to-suite
conversion, including validation against the expected schema structure.

Supports two case formats:
1. Content assertions: classic prompt + structural assertions on response text
2. Trajectory contracts: prompt + trajectory assertions + argument contracts

The YAML format is validated on parse, and errors include the line number
and a plain-English description of what is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from callspec.assertions.base import BaseAssertion
from callspec.assertions.structural import (
    ContainsKeys,
    DoesNotContain,
    EndsWith,
    IsValidJson,
    LengthBetween,
    MatchesPattern,
    MatchesSchema,
    StartsWith,
)
from callspec.core.config import CallspecConfig
from callspec.core.suite import AssertionCase, AssertionSuite
from callspec.core.types import Severity
from callspec.errors import SuiteParseError

# Current YAML suite schema version
SUITE_SCHEMA_VERSION = "1.0"

# Maps YAML assertion type names to their builder functions.
# Each function takes a params dict and returns a BaseAssertion instance.
_ASSERTION_BUILDERS: dict[str, Any] = {}


def _register_assertion(name: str):
    """Decorator to register a YAML assertion type builder."""
    def decorator(fn):
        _ASSERTION_BUILDERS[name] = fn
        return fn
    return decorator


# -- Structural assertion builders --

@_register_assertion("is_valid_json")
def _build_is_valid_json(params: dict[str, Any]) -> BaseAssertion:
    return IsValidJson()


@_register_assertion("matches_schema")
def _build_matches_schema(params: dict[str, Any]) -> BaseAssertion:
    schema = params.get("schema")
    if not schema:
        raise ValueError("matches_schema requires a 'schema' parameter")
    return MatchesSchema(schema)


@_register_assertion("contains_keys")
def _build_contains_keys(params: dict[str, Any]) -> BaseAssertion:
    keys = params.get("keys")
    if not keys:
        raise ValueError("contains_keys requires a 'keys' parameter (list of strings)")
    return ContainsKeys(keys)


@_register_assertion("length_between")
def _build_length_between(params: dict[str, Any]) -> BaseAssertion:
    min_chars = params.get("min_chars", 0)
    max_chars = params.get("max_chars")
    if max_chars is None:
        raise ValueError("length_between requires 'max_chars' parameter")
    return LengthBetween(min_chars, max_chars)


@_register_assertion("matches_pattern")
def _build_matches_pattern(params: dict[str, Any]) -> BaseAssertion:
    pattern = params.get("pattern")
    if not pattern:
        raise ValueError("matches_pattern requires a 'pattern' parameter")
    return MatchesPattern(pattern)


@_register_assertion("does_not_contain")
def _build_does_not_contain(params: dict[str, Any]) -> BaseAssertion:
    text = params.get("text", params.get("pattern", ""))
    is_regex = params.get("is_regex", False)
    if not text:
        raise ValueError("does_not_contain requires a 'text' or 'pattern' parameter")
    return DoesNotContain(text, is_regex)


@_register_assertion("starts_with")
def _build_starts_with(params: dict[str, Any]) -> BaseAssertion:
    prefix = params.get("prefix", "")
    if not prefix:
        raise ValueError("starts_with requires a 'prefix' parameter")
    return StartsWith(prefix)


@_register_assertion("ends_with")
def _build_ends_with(params: dict[str, Any]) -> BaseAssertion:
    suffix = params.get("suffix", "")
    if not suffix:
        raise ValueError("ends_with requires a 'suffix' parameter")
    return EndsWith(suffix)


# -- Trajectory assertion builders --
# These produce TrajectoryAssertion instances from the YAML `trajectory` section.

from callspec.assertions.trajectory_base import TrajectoryAssertion
from callspec.assertions.trajectory import (
    CallCount,
    CallsExactly,
    CallsSubset,
    CallsTool,
    CallsToolsInOrder,
    DoesNotCall,
    NoRepeatedCalls,
)
from callspec.assertions.contract import (
    ArgumentContainsKey,
    ArgumentMatchesPattern,
    ArgumentMatchesSchema,
    ArgumentNotEmpty,
    ArgumentValueIn,
)

# Maps trajectory YAML keys to builders returning TrajectoryAssertion
_TRAJECTORY_BUILDERS: dict[str, Any] = {}


def _register_trajectory(name: str):
    """Decorator to register a YAML trajectory assertion builder."""
    def decorator(fn):
        _TRAJECTORY_BUILDERS[name] = fn
        return fn
    return decorator


@_register_trajectory("calls_tool")
def _build_calls_tool(params: Any) -> TrajectoryAssertion:
    if isinstance(params, str):
        return CallsTool(params)
    raise ValueError("calls_tool expects a tool name string")


@_register_trajectory("calls_tools_in_order")
def _build_calls_tools_in_order(params: Any) -> TrajectoryAssertion:
    if isinstance(params, list):
        return CallsToolsInOrder(params)
    raise ValueError("calls_tools_in_order expects a list of tool names")


@_register_trajectory("calls_exactly")
def _build_calls_exactly(params: Any) -> TrajectoryAssertion:
    if isinstance(params, list):
        return CallsExactly(params)
    raise ValueError("calls_exactly expects a list of tool names")


@_register_trajectory("calls_subset")
def _build_calls_subset(params: Any) -> TrajectoryAssertion:
    if isinstance(params, list):
        return CallsSubset(params)
    raise ValueError("calls_subset expects a list of tool names")


@_register_trajectory("does_not_call")
def _build_does_not_call_traj(params: Any) -> TrajectoryAssertion:
    if isinstance(params, str):
        return DoesNotCall(params)
    raise ValueError("does_not_call expects a tool name string")


@_register_trajectory("no_repeated_calls")
def _build_no_repeated(params: Any) -> TrajectoryAssertion:
    if isinstance(params, str):
        return NoRepeatedCalls(params)
    raise ValueError("no_repeated_calls expects a tool name string")


@_register_trajectory("call_count")
def _build_call_count(params: Any) -> TrajectoryAssertion:
    if not isinstance(params, dict):
        raise ValueError("call_count expects a dict with tool_name, min_count, max_count")
    tool_name = params.get("tool_name")
    if not tool_name:
        raise ValueError("call_count requires 'tool_name'")
    return CallCount(
        tool_name=tool_name,
        min_count=params.get("min_count", 0),
        max_count=params.get("max_count"),
    )


def _build_trajectory_assertions(
    trajectory_defs: list[dict[str, Any]], filepath: str
) -> list[TrajectoryAssertion]:
    """Parse the `trajectory` section of a YAML case into TrajectoryAssertion list."""
    assertions: list[TrajectoryAssertion] = []

    for entry in trajectory_defs:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise SuiteParseError(
                filepath,
                f"Each trajectory entry must be a single-key dict, got: {entry}. "
                f"Available types: {sorted(_TRAJECTORY_BUILDERS.keys())}",
            )
        key, value = next(iter(entry.items()))
        builder = _TRAJECTORY_BUILDERS.get(key)
        if builder is None:
            raise SuiteParseError(
                filepath,
                f"Unknown trajectory assertion '{key}'. "
                f"Available: {sorted(_TRAJECTORY_BUILDERS.keys())}",
            )
        try:
            assertions.append(builder(value))
        except (ValueError, TypeError) as build_error:
            raise SuiteParseError(
                filepath,
                f"Invalid parameters for trajectory assertion '{key}': {build_error}",
            ) from build_error

    return assertions


def _build_contract_assertions(
    contracts_def: dict[str, list[dict[str, Any]]], filepath: str
) -> list[TrajectoryAssertion]:
    """Parse the `contracts` section of a YAML case into TrajectoryAssertion list.

    contracts_def is a dict mapping tool_name to a list of constraint dicts:
      contracts:
        search_flights:
          - key: "origin"
            not_empty: true
          - key: "destination"
            matches_pattern: "^[A-Z]{3}$"
        book_flight:
          - schema: { ... }
    """
    assertions: list[TrajectoryAssertion] = []

    for tool_name, constraints in contracts_def.items():
        if not isinstance(constraints, list):
            raise SuiteParseError(
                filepath,
                f"Contracts for '{tool_name}' must be a list of constraint dicts.",
            )
        for constraint in constraints:
            assertions.extend(
                _build_single_contract(tool_name, constraint, filepath)
            )

    return assertions


def _build_single_contract(
    tool_name: str, constraint: dict[str, Any], filepath: str
) -> list[TrajectoryAssertion]:
    """Build one or more contract assertions from a single constraint dict."""
    results: list[TrajectoryAssertion] = []
    key = constraint.get("key")

    if "schema" in constraint:
        results.append(ArgumentMatchesSchema(tool_name, constraint["schema"]))

    if key:
        if constraint.get("not_empty"):
            results.append(ArgumentNotEmpty(tool_name, key))

        if "matches_pattern" in constraint:
            results.append(
                ArgumentMatchesPattern(tool_name, key, constraint["matches_pattern"])
            )

        if "value_in" in constraint:
            results.append(
                ArgumentValueIn(tool_name, key, constraint["value_in"])
            )

        if "contains_key" in constraint:
            # For nested key checking: the key at this level is the arg name,
            # contains_key verifies it exists at all
            results.append(ArgumentContainsKey(tool_name, key))

    if not results:
        raise SuiteParseError(
            filepath,
            f"Contract for '{tool_name}' has no recognizable constraints: {constraint}. "
            f"Use: not_empty, matches_pattern, value_in, schema, contains_key.",
        )

    return results


def _build_assertion(assertion_def: dict[str, Any], filepath: str) -> BaseAssertion:
    """Convert a single YAML assertion definition to a BaseAssertion instance."""
    assertion_type = assertion_def.get("type")
    if not assertion_type:
        raise SuiteParseError(
            filepath,
            "Each assertion must have a 'type' field. "
            f"Available types: {sorted(_ASSERTION_BUILDERS.keys())}",
        )

    builder = _ASSERTION_BUILDERS.get(assertion_type)
    if builder is None:
        raise SuiteParseError(
            filepath,
            f"Unknown assertion type '{assertion_type}'. "
            f"Available types: {sorted(_ASSERTION_BUILDERS.keys())}",
        )

    params = assertion_def.get("params", {})
    try:
        return builder(params)
    except (ValueError, TypeError) as build_error:
        raise SuiteParseError(
            filepath,
            f"Invalid parameters for assertion '{assertion_type}': {build_error}",
        ) from build_error


def _parse_config(raw_config: dict[str, Any]) -> CallspecConfig:
    """Build a CallspecConfig from YAML config section."""
    known_fields = {f.name for f in CallspecConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw_config.items() if k in known_fields}
    return CallspecConfig(**filtered)


def _parse_case(case_def: dict[str, Any], filepath: str, index: int) -> AssertionCase:
    """Parse a single case definition from the YAML cases list.

    Supports three assertion source sections:
    - assertions: classic content assertions on response text
    - trajectory: tool-call sequence assertions
    - contracts: per-tool argument constraints
    """
    name = case_def.get("name")
    if not name:
        raise SuiteParseError(
            filepath,
            f"Case at index {index} is missing a 'name' field. "
            f"Every case must have a unique name for report output.",
        )

    prompt = case_def.get("prompt", "")
    messages = case_def.get("messages")

    if not prompt and not messages:
        raise SuiteParseError(
            filepath,
            f"Case '{name}' has neither 'prompt' nor 'messages'. "
            f"At least one input source is required.",
        )

    # Content assertions (classic text-based)
    raw_assertions = case_def.get("assertions", [])
    parsed_assertions = [
        _build_assertion(a, filepath) for a in raw_assertions
    ]

    # Trajectory assertions (tool call sequence)
    raw_trajectory = case_def.get("trajectory", [])
    trajectory_assertions: list[TrajectoryAssertion] = []
    if raw_trajectory:
        trajectory_assertions.extend(
            _build_trajectory_assertions(raw_trajectory, filepath)
        )

    # Contract assertions (per-tool argument constraints)
    raw_contracts = case_def.get("contracts", {})
    if raw_contracts:
        trajectory_assertions.extend(
            _build_contract_assertions(raw_contracts, filepath)
        )

    # At least one assertion type is required
    if not parsed_assertions and not trajectory_assertions:
        raise SuiteParseError(
            filepath,
            f"Case '{name}' has no assertions, trajectory checks, or contracts. "
            f"Add at least one to make the case meaningful.",
        )

    severity_str = case_def.get("severity", "error").lower()
    severity = Severity.WARNING if severity_str == "warning" else Severity.ERROR

    return AssertionCase(
        name=name,
        prompt=prompt,
        assertions=parsed_assertions,
        trajectory_assertions=trajectory_assertions,
        messages=messages,
        severity=severity,
    )


def load_yaml_suite(filepath: str | Path) -> AssertionSuite:
    """Parse a YAML file into an AssertionSuite.

    Validates structure on parse. Errors include the filepath and a
    plain-English description of what is wrong.

    Expected top-level keys:
        version  - schema version (currently "1.0")
        name     - suite name (optional, defaults to filename)
        provider - provider config (name, model, etc.)
        config   - CallspecConfig overrides
        cases    - ordered list of test cases
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise SuiteParseError(str(filepath), f"Suite file not found: {filepath}")

    try:
        raw = yaml.safe_load(filepath.read_text())
    except yaml.YAMLError as yaml_error:
        # Extract line number from PyYAML error when available
        line_number = None
        if hasattr(yaml_error, "problem_mark") and yaml_error.problem_mark:
            line_number = yaml_error.problem_mark.line + 1
        raise SuiteParseError(
            str(filepath),
            f"YAML parse error: {yaml_error}",
            line_number=line_number,
        ) from yaml_error

    if not isinstance(raw, dict):
        raise SuiteParseError(
            str(filepath),
            "Suite file must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}.",
        )

    # Version check
    version = str(raw.get("version", "1.0"))
    if not version.startswith("1."):
        raise SuiteParseError(
            str(filepath),
            f"Unsupported suite schema version '{version}'. "
            f"This version of Callspec supports version 1.x.",
        )

    suite_name = raw.get("name", filepath.stem)

    # Config
    raw_config = raw.get("config", {})
    config = _parse_config(raw_config) if raw_config else CallspecConfig()

    # Cases
    raw_cases = raw.get("cases", [])
    if not raw_cases:
        raise SuiteParseError(
            str(filepath),
            "Suite has no cases. Add at least one case with a prompt and assertions.",
        )

    cases = [
        _parse_case(case_def, str(filepath), index)
        for index, case_def in enumerate(raw_cases)
    ]

    return AssertionSuite(
        name=suite_name,
        cases=cases,
        config=config,
    )
