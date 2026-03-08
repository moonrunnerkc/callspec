"""Structural assertions: deterministic checks on response form.

All assertions in this module operate on the string content of a provider
response. They never call an LLM or compute embeddings. They are the cheapest
assertions to run, and every LLM application that expects structured output
should have them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

import jsonschema

from verdict.assertions.base import BaseAssertion
from verdict.core.config import VerdictConfig
from verdict.core.types import IndividualAssertionResult


class IsValidJson(BaseAssertion):
    """Passes if the content parses as valid JSON."""

    assertion_type = "structural"
    assertion_name = "is_valid_json"

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        try:
            json.loads(content)
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message="Response is valid JSON.",
            )
        except json.JSONDecodeError as parse_error:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Response is not valid JSON. "
                    f"Parse error at position {parse_error.pos}: {parse_error.msg}"
                ),
                details={
                    "error": parse_error.msg,
                    "position": parse_error.pos,
                    "line": parse_error.lineno,
                    "column": parse_error.colno,
                },
            )


class MatchesSchema(BaseAssertion):
    """Passes if the content parses as JSON and validates against a JSON Schema."""

    assertion_type = "structural"
    assertion_name = "matches_schema"

    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as parse_error:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot validate schema: response is not valid JSON. "
                    f"Parse error at position {parse_error.pos}: {parse_error.msg}"
                ),
                details={"error": "json_parse_failure", "position": parse_error.pos},
            )

        validator = jsonschema.Draft7Validator(self._schema)
        violations = list(validator.iter_errors(parsed))

        if not violations:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message="Response matches the provided JSON Schema.",
            )

        # Collect all violations with their JSON paths for precise debugging
        violation_details = [
            {
                "path": list(violation.absolute_path),
                "message": violation.message,
                "validator": violation.validator,
            }
            for violation in violations
        ]

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Response JSON has {len(violations)} schema violation"
                f"{'s' if len(violations) != 1 else ''}: "
                f"{violations[0].message}"
            ),
            details={"violations": violation_details, "violation_count": len(violations)},
        )


class ContainsKeys(BaseAssertion):
    """Passes if the response JSON contains all specified top-level keys."""

    assertion_type = "structural"
    assertion_name = "contains_keys"

    def __init__(self, keys: Sequence[str]) -> None:
        self._keys = list(keys)

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as parse_error:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot check keys: response is not valid JSON. "
                    f"Parse error at position {parse_error.pos}: {parse_error.msg}"
                ),
                details={"error": "json_parse_failure"},
            )

        if not isinstance(parsed, dict):
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=False,
                message=(
                    f"Cannot check keys: response JSON is {type(parsed).__name__}, "
                    f"not an object. Top-level key checking requires a JSON object."
                ),
                details={"actual_type": type(parsed).__name__},
            )

        missing_keys = [key for key in self._keys if key not in parsed]

        if not missing_keys:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=f"Response contains all {len(self._keys)} required keys.",
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Response is missing {len(missing_keys)} required key"
                f"{'s' if len(missing_keys) != 1 else ''}: {missing_keys}. "
                f"Present keys: {list(parsed.keys())}"
            ),
            details={
                "missing_keys": missing_keys,
                "present_keys": list(parsed.keys()),
                "required_keys": self._keys,
            },
        )


class LengthBetween(BaseAssertion):
    """Passes if content length in characters falls within [min_chars, max_chars]."""

    assertion_type = "structural"
    assertion_name = "length_between"

    def __init__(self, min_chars: int, max_chars: int) -> None:
        if min_chars < 0:
            raise ValueError(f"min_chars must be non-negative, got {min_chars}")
        if max_chars < min_chars:
            raise ValueError(
                f"max_chars ({max_chars}) must be >= min_chars ({min_chars})"
            )
        self._min_chars = min_chars
        self._max_chars = max_chars

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        actual_length = len(content)
        within_bounds = self._min_chars <= actual_length <= self._max_chars

        if within_bounds:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Response length {actual_length} chars is within "
                    f"[{self._min_chars}, {self._max_chars}]."
                ),
                details={"length": actual_length},
            )

        direction = "below" if actual_length < self._min_chars else "above"
        bound = self._min_chars if direction == "below" else self._max_chars

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Response length {actual_length} chars is {direction} "
                f"the {'minimum' if direction == 'below' else 'maximum'} of {bound}. "
                f"Allowed range: [{self._min_chars}, {self._max_chars}]."
            ),
            details={
                "length": actual_length,
                "min_chars": self._min_chars,
                "max_chars": self._max_chars,
            },
        )


class MatchesPattern(BaseAssertion):
    """Passes if the content matches a regular expression.

    Named groups in the regex are captured into the result details,
    enabling extraction of structured data from freeform responses.
    """

    assertion_type = "structural"
    assertion_name = "matches_pattern"

    def __init__(self, pattern: str) -> None:
        self._pattern_str = pattern
        try:
            self._pattern = re.compile(pattern, re.DOTALL)
        except re.error as regex_error:
            raise ValueError(
                f"Invalid regex pattern '{pattern}': {regex_error}"
            ) from regex_error

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        match = self._pattern.search(content)

        if match:
            named_groups = {
                name: value
                for name, value in match.groupdict().items()
                if value is not None
            }
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=f"Response matches pattern '{self._pattern_str}'.",
                details={"matched_groups": named_groups} if named_groups else {},
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Response does not match pattern '{self._pattern_str}'. "
                f"Content length: {len(content)} chars."
            ),
            details={"pattern": self._pattern_str, "content_length": len(content)},
        )


class DoesNotContain(BaseAssertion):
    """Passes if the specified text or pattern does not appear in the content.

    Primary assertion for negative content checks: confirming the model
    did not output a competitor name, prohibited phrase, or unwanted format.
    """

    assertion_type = "structural"
    assertion_name = "does_not_contain"

    def __init__(self, text_or_pattern: str, is_regex: bool = False) -> None:
        self._text_or_pattern = text_or_pattern
        self._is_regex = is_regex
        if is_regex:
            try:
                self._compiled = re.compile(text_or_pattern, re.DOTALL)
            except re.error as regex_error:
                raise ValueError(
                    f"Invalid regex pattern '{text_or_pattern}': {regex_error}"
                ) from regex_error
        else:
            self._compiled = None

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        if self._is_regex and self._compiled is not None:
            found = self._compiled.search(content) is not None
        else:
            found = self._text_or_pattern in content

        if not found:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=(
                    f"Response does not contain "
                    f"{'pattern' if self._is_regex else 'text'} "
                    f"'{self._text_or_pattern}'."
                ),
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Response contains prohibited "
                f"{'pattern' if self._is_regex else 'text'} "
                f"'{self._text_or_pattern}'. "
                f"Content length: {len(content)} chars."
            ),
            details={
                "prohibited": self._text_or_pattern,
                "is_regex": self._is_regex,
            },
        )


class StartsWith(BaseAssertion):
    """Passes if the response content starts with the specified prefix."""

    assertion_type = "structural"
    assertion_name = "starts_with"

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        if content.startswith(self._prefix):
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=f"Response starts with '{self._prefix}'.",
            )

        # Show what the content actually starts with for quick debugging
        actual_start = content[: len(self._prefix) + 20]
        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Response does not start with '{self._prefix}'. "
                f"Actual start: '{actual_start}...'"
            ),
            details={"expected_prefix": self._prefix, "actual_start": actual_start},
        )


class EndsWith(BaseAssertion):
    """Passes if the response content ends with the specified suffix."""

    assertion_type = "structural"
    assertion_name = "ends_with"

    def __init__(self, suffix: str) -> None:
        self._suffix = suffix

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        if content.endswith(self._suffix):
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message=f"Response ends with '{self._suffix}'.",
            )

        actual_end = content[-(len(self._suffix) + 20) :]
        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=False,
            message=(
                f"Response does not end with '{self._suffix}'. "
                f"Actual end: '...{actual_end}'"
            ),
            details={"expected_suffix": self._suffix, "actual_end": actual_end},
        )
