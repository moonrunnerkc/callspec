"""YAML suite parser: loads assertion suites from YAML configuration files.

Both the Python API and the YAML format compile to the same internal
representation (AssertionSuite). This module handles the YAML-to-suite
conversion, including validation against the expected schema structure.

The YAML format is validated on parse, and errors include the line number
and a plain-English description of what is wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from verdict.assertions.base import BaseAssertion
from verdict.assertions.structural import (
    ContainsKeys,
    DoesNotContain,
    EndsWith,
    IsValidJson,
    LengthBetween,
    MatchesPattern,
    MatchesSchema,
    StartsWith,
)
from verdict.core.config import VerdictConfig
from verdict.core.suite import AssertionCase, AssertionSuite
from verdict.core.types import Severity
from verdict.errors import SuiteParseError

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


# -- Semantic assertion builders (lazy import to avoid requiring verdict[semantic]) --

@_register_assertion("semantic_intent_matches")
def _build_semantic_intent_matches(params: dict[str, Any]) -> BaseAssertion:
    from verdict.assertions.semantic import SemanticIntentMatches
    reference = params.get("reference_intent", params.get("reference", ""))
    threshold = params.get("threshold")
    if not reference:
        raise ValueError("semantic_intent_matches requires a 'reference_intent' parameter")
    return SemanticIntentMatches(reference, threshold)


@_register_assertion("does_not_discuss")
def _build_does_not_discuss(params: dict[str, Any]) -> BaseAssertion:
    from verdict.assertions.semantic import DoesNotDiscuss
    topic = params.get("topic", "")
    threshold = params.get("threshold")
    if not topic:
        raise ValueError("does_not_discuss requires a 'topic' parameter")
    return DoesNotDiscuss(topic, threshold)


@_register_assertion("is_factually_consistent_with")
def _build_is_factually_consistent_with(params: dict[str, Any]) -> BaseAssertion:
    from verdict.assertions.semantic import IsFactuallyConsistentWith
    reference = params.get("reference_text", params.get("reference", ""))
    threshold = params.get("threshold")
    if not reference:
        raise ValueError("is_factually_consistent_with requires a 'reference_text' parameter")
    return IsFactuallyConsistentWith(reference, threshold)


@_register_assertion("uses_language_at_grade_level")
def _build_uses_language_at_grade_level(params: dict[str, Any]) -> BaseAssertion:
    from verdict.assertions.semantic import UsesLanguageAtGradeLevel
    grade = params.get("grade")
    tolerance = params.get("tolerance", 2)
    if grade is None:
        raise ValueError("uses_language_at_grade_level requires a 'grade' parameter")
    return UsesLanguageAtGradeLevel(grade, tolerance)


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


def _parse_config(raw_config: dict[str, Any]) -> VerdictConfig:
    """Build a VerdictConfig from YAML config section."""
    known_fields = {f.name for f in VerdictConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in raw_config.items() if k in known_fields}
    return VerdictConfig(**filtered)


def _parse_case(case_def: dict[str, Any], filepath: str, index: int) -> AssertionCase:
    """Parse a single case definition from the YAML cases list."""
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

    raw_assertions = case_def.get("assertions", [])
    if not raw_assertions:
        raise SuiteParseError(
            filepath,
            f"Case '{name}' has no assertions. "
            f"Add at least one assertion to make the case meaningful.",
        )

    parsed_assertions = [
        _build_assertion(a, filepath) for a in raw_assertions
    ]

    severity_str = case_def.get("severity", "error").lower()
    severity = Severity.WARNING if severity_str == "warning" else Severity.ERROR

    return AssertionCase(
        name=name,
        prompt=prompt,
        assertions=parsed_assertions,
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
        config   - VerdictConfig overrides
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
            f"This version of Verdict supports version 1.x.",
        )

    suite_name = raw.get("name", filepath.stem)

    # Config
    raw_config = raw.get("config", {})
    config = _parse_config(raw_config) if raw_config else VerdictConfig()

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
