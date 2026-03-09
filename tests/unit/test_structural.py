"""Unit tests for all structural assertion types.

Each assertion is tested with clear pass/fail cases and edge conditions.
Tests use CallspecConfig defaults unless testing config-dependent behavior.
"""

from __future__ import annotations

import json

import pytest

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

CONFIG = CallspecConfig()


# -- IsValidJson --


class TestIsValidJson:

    def test_valid_json_object(self) -> None:
        content = '{"title": "Test", "count": 42}'
        assertion_result = IsValidJson().evaluate(content, CONFIG)
        assert assertion_result.passed is True
        assert assertion_result.assertion_type == "structural"
        assert assertion_result.assertion_name == "is_valid_json"

    def test_valid_json_array(self) -> None:
        content = '[1, 2, 3]'
        assertion_result = IsValidJson().evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_valid_json_string(self) -> None:
        content = '"just a string"'
        assertion_result = IsValidJson().evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_valid_json_number(self) -> None:
        assertion_result = IsValidJson().evaluate("42", CONFIG)
        assert assertion_result.passed is True

    def test_valid_json_boolean(self) -> None:
        assertion_result = IsValidJson().evaluate("true", CONFIG)
        assert assertion_result.passed is True

    def test_valid_json_null(self) -> None:
        assertion_result = IsValidJson().evaluate("null", CONFIG)
        assert assertion_result.passed is True

    def test_invalid_json_plain_text(self) -> None:
        assertion_result = IsValidJson().evaluate("not json at all", CONFIG)
        assert assertion_result.passed is False
        assert "not valid JSON" in assertion_result.message
        assert "position" in assertion_result.details

    def test_invalid_json_truncated(self) -> None:
        content = '{"title": "incomplete'
        assertion_result = IsValidJson().evaluate(content, CONFIG)
        assert assertion_result.passed is False

    def test_empty_string(self) -> None:
        assertion_result = IsValidJson().evaluate("", CONFIG)
        assert assertion_result.passed is False

    def test_nested_json(self) -> None:
        content = '{"outer": {"inner": [1, 2, {"deep": true}]}}'
        assertion_result = IsValidJson().evaluate(content, CONFIG)
        assert assertion_result.passed is True


# -- MatchesSchema --


class TestMatchesSchema:

    ARTICLE_SCHEMA = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "word_count": {"type": "integer"},
        },
        "required": ["title", "summary"],
    }

    def test_valid_schema_match(self) -> None:
        content = json.dumps({"title": "Test", "summary": "A summary", "word_count": 100})
        assertion_result = MatchesSchema(self.ARTICLE_SCHEMA).evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_valid_with_extra_keys(self) -> None:
        content = json.dumps({"title": "T", "summary": "S", "extra": "allowed"})
        assertion_result = MatchesSchema(self.ARTICLE_SCHEMA).evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_missing_required_key(self) -> None:
        content = json.dumps({"title": "T"})
        assertion_result = MatchesSchema(self.ARTICLE_SCHEMA).evaluate(content, CONFIG)
        assert assertion_result.passed is False
        assert "schema violation" in assertion_result.message
        assert assertion_result.details["violation_count"] == 1

    def test_wrong_type(self) -> None:
        content = json.dumps({"title": "T", "summary": "S", "word_count": "not-an-int"})
        assertion_result = MatchesSchema(self.ARTICLE_SCHEMA).evaluate(content, CONFIG)
        assert assertion_result.passed is False

    def test_not_json(self) -> None:
        assertion_result = MatchesSchema(self.ARTICLE_SCHEMA).evaluate("not json", CONFIG)
        assert assertion_result.passed is False
        assert "not valid JSON" in assertion_result.message

    def test_multiple_violations(self) -> None:
        # Missing both required fields
        content = json.dumps({"word_count": "wrong"})
        assertion_result = MatchesSchema(self.ARTICLE_SCHEMA).evaluate(content, CONFIG)
        assert assertion_result.passed is False
        assert assertion_result.details["violation_count"] >= 2


# -- ContainsKeys --


class TestContainsKeys:

    def test_all_keys_present(self) -> None:
        content = json.dumps({"title": "T", "summary": "S", "tags": []})
        assertion_result = ContainsKeys(["title", "summary"]).evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_missing_one_key(self) -> None:
        content = json.dumps({"title": "T"})
        assertion_result = ContainsKeys(["title", "summary"]).evaluate(content, CONFIG)
        assert assertion_result.passed is False
        assert "summary" in str(assertion_result.details["missing_keys"])

    def test_missing_all_keys(self) -> None:
        content = json.dumps({"other": "value"})
        assertion_result = ContainsKeys(["title", "summary"]).evaluate(content, CONFIG)
        assert assertion_result.passed is False
        assert len(assertion_result.details["missing_keys"]) == 2

    def test_not_json(self) -> None:
        assertion_result = ContainsKeys(["title"]).evaluate("not json", CONFIG)
        assert assertion_result.passed is False

    def test_json_array_not_object(self) -> None:
        assertion_result = ContainsKeys(["title"]).evaluate("[1, 2, 3]", CONFIG)
        assert assertion_result.passed is False
        assert "not an object" in assertion_result.message

    def test_empty_keys_list(self) -> None:
        content = json.dumps({"anything": "here"})
        assertion_result = ContainsKeys([]).evaluate(content, CONFIG)
        assert assertion_result.passed is True


# -- LengthBetween --


class TestLengthBetween:

    def test_within_range(self) -> None:
        content = "Hello, world!"  # 13 chars
        assertion_result = LengthBetween(10, 20).evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_at_minimum(self) -> None:
        content = "x" * 10
        assertion_result = LengthBetween(10, 20).evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_at_maximum(self) -> None:
        content = "x" * 20
        assertion_result = LengthBetween(10, 20).evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_below_minimum(self) -> None:
        content = "short"
        assertion_result = LengthBetween(10, 20).evaluate(content, CONFIG)
        assert assertion_result.passed is False
        assert "below" in assertion_result.message

    def test_above_maximum(self) -> None:
        content = "x" * 100
        assertion_result = LengthBetween(10, 20).evaluate(content, CONFIG)
        assert assertion_result.passed is False
        assert "above" in assertion_result.message

    def test_empty_string_with_zero_min(self) -> None:
        assertion_result = LengthBetween(0, 100).evaluate("", CONFIG)
        assert assertion_result.passed is True

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            LengthBetween(20, 10)

    def test_negative_min_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            LengthBetween(-1, 10)


# -- MatchesPattern --


class TestMatchesPattern:

    def test_simple_match(self) -> None:
        assertion_result = MatchesPattern(r"\d+").evaluate("answer is 42", CONFIG)
        assert assertion_result.passed is True

    def test_no_match(self) -> None:
        assertion_result = MatchesPattern(r"^\d+$").evaluate("no digits here", CONFIG)
        assert assertion_result.passed is False

    def test_named_groups_captured(self) -> None:
        pattern = r"name: (?P<name>\w+), age: (?P<age>\d+)"
        content = "name: Alice, age: 30"
        assertion_result = MatchesPattern(pattern).evaluate(content, CONFIG)
        assert assertion_result.passed is True
        assert assertion_result.details["matched_groups"]["name"] == "Alice"
        assert assertion_result.details["matched_groups"]["age"] == "30"

    def test_multiline_with_dotall(self) -> None:
        content = "line one\nline two\nline three"
        assertion_result = MatchesPattern(r"one.*three").evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid regex"):
            MatchesPattern(r"[invalid")

    def test_empty_content(self) -> None:
        assertion_result = MatchesPattern(r".+").evaluate("", CONFIG)
        assert assertion_result.passed is False

    def test_full_string_anchor(self) -> None:
        assertion_result = MatchesPattern(r"^exact$").evaluate("exact", CONFIG)
        assert assertion_result.passed is True
        assertion_result = MatchesPattern(r"^exact$").evaluate("not exact", CONFIG)
        assert assertion_result.passed is False


# -- DoesNotContain --


class TestDoesNotContain:

    def test_text_absent(self) -> None:
        assertion_result = DoesNotContain("forbidden").evaluate("safe content", CONFIG)
        assert assertion_result.passed is True

    def test_text_present(self) -> None:
        assertion_result = DoesNotContain("forbidden").evaluate(
            "this is forbidden content", CONFIG
        )
        assert assertion_result.passed is False
        assert "prohibited" in assertion_result.message

    def test_regex_absent(self) -> None:
        assertion_result = DoesNotContain(r"\d{3}-\d{4}", is_regex=True).evaluate(
            "no phone numbers here", CONFIG
        )
        assert assertion_result.passed is True

    def test_regex_present(self) -> None:
        assertion_result = DoesNotContain(r"\d{3}-\d{4}", is_regex=True).evaluate(
            "call 555-1234", CONFIG
        )
        assert assertion_result.passed is False

    def test_case_sensitive(self) -> None:
        assertion_result = DoesNotContain("Forbidden").evaluate("forbidden", CONFIG)
        assert assertion_result.passed is True

    def test_invalid_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid regex"):
            DoesNotContain(r"[bad", is_regex=True)

    def test_empty_content(self) -> None:
        assertion_result = DoesNotContain("anything").evaluate("", CONFIG)
        assert assertion_result.passed is True


# -- StartsWith --


class TestStartsWith:

    def test_starts_with_match(self) -> None:
        assertion_result = StartsWith("Hello").evaluate("Hello, world!", CONFIG)
        assert assertion_result.passed is True

    def test_starts_with_no_match(self) -> None:
        assertion_result = StartsWith("Goodbye").evaluate("Hello, world!", CONFIG)
        assert assertion_result.passed is False
        assert "does not start with" in assertion_result.message

    def test_exact_match(self) -> None:
        assertion_result = StartsWith("exact").evaluate("exact", CONFIG)
        assert assertion_result.passed is True

    def test_empty_prefix_always_passes(self) -> None:
        assertion_result = StartsWith("").evaluate("anything", CONFIG)
        assert assertion_result.passed is True

    def test_prefix_longer_than_content(self) -> None:
        assertion_result = StartsWith("very long prefix").evaluate("short", CONFIG)
        assert assertion_result.passed is False


# -- EndsWith --


class TestEndsWith:

    def test_ends_with_match(self) -> None:
        assertion_result = EndsWith("world!").evaluate("Hello, world!", CONFIG)
        assert assertion_result.passed is True

    def test_ends_with_no_match(self) -> None:
        assertion_result = EndsWith("world").evaluate("Hello, world!", CONFIG)
        assert assertion_result.passed is False

    def test_json_ending(self) -> None:
        content = '{"key": "value"}'
        assertion_result = EndsWith("}").evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_empty_suffix_always_passes(self) -> None:
        assertion_result = EndsWith("").evaluate("anything", CONFIG)
        assert assertion_result.passed is True

    def test_suffix_longer_than_content(self) -> None:
        assertion_result = EndsWith("very long suffix").evaluate("short", CONFIG)
        assert assertion_result.passed is False
