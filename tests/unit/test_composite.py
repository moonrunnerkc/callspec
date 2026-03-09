"""Unit tests for composite assertions: NegationWrapper, AndAssertion, OrAssertion."""

from __future__ import annotations

from callspec.assertions.composite import AndAssertion, NegationWrapper, OrAssertion
from callspec.assertions.structural import IsValidJson, LengthBetween, StartsWith
from callspec.core.config import CallspecConfig

CONFIG = CallspecConfig()


class TestNegationWrapper:

    def test_negates_passing(self) -> None:
        inner = IsValidJson()
        negated = NegationWrapper(inner)
        # Valid JSON passes inner, so negation should fail
        assertion_result = negated.evaluate('{"valid": true}', CONFIG)
        assert assertion_result.passed is False

    def test_negates_failing(self) -> None:
        inner = IsValidJson()
        negated = NegationWrapper(inner)
        # Invalid JSON fails inner, so negation should pass
        assertion_result = negated.evaluate("not json", CONFIG)
        assert assertion_result.passed is True

    def test_assertion_name_includes_inner(self) -> None:
        inner = IsValidJson()
        negated = NegationWrapper(inner)
        assert "is_valid_json" in negated.assertion_name


class TestAndAssertion:

    def test_all_pass(self) -> None:
        combined = AndAssertion([
            IsValidJson(),
            LengthBetween(1, 100),
        ])
        assertion_result = combined.evaluate('{"ok": true}', CONFIG)
        assert assertion_result.passed is True

    def test_one_fails(self) -> None:
        combined = AndAssertion([
            IsValidJson(),
            StartsWith("WRONG"),
        ])
        assertion_result = combined.evaluate('{"ok": true}', CONFIG)
        assert assertion_result.passed is False
        assert "1 of 2" in assertion_result.message

    def test_all_fail(self) -> None:
        combined = AndAssertion([
            IsValidJson(),
            StartsWith("prefix"),
        ])
        assertion_result = combined.evaluate("not json and no prefix", CONFIG)
        assert assertion_result.passed is False
        assert "2 of 2" in assertion_result.message

    def test_empty_and_passes(self) -> None:
        # Vacuous truth: AND over an empty set passes
        combined = AndAssertion([])
        assertion_result = combined.evaluate("anything", CONFIG)
        assert assertion_result.passed is True


class TestOrAssertion:

    def test_all_pass(self) -> None:
        combined = OrAssertion([
            IsValidJson(),
            LengthBetween(1, 100),
        ])
        assertion_result = combined.evaluate('{"ok": true}', CONFIG)
        assert assertion_result.passed is True

    def test_one_passes(self) -> None:
        combined = OrAssertion([
            IsValidJson(),  # fails
            LengthBetween(1, 100),  # passes
        ])
        assertion_result = combined.evaluate("not json but short", CONFIG)
        assert assertion_result.passed is True

    def test_none_pass(self) -> None:
        combined = OrAssertion([
            IsValidJson(),  # fails
            LengthBetween(1000, 2000),  # fails
        ])
        assertion_result = combined.evaluate("short non-json", CONFIG)
        assert assertion_result.passed is False

    def test_empty_or_fails(self) -> None:
        # OR over empty set: no assertion passes, so result is False
        combined = OrAssertion([])
        assertion_result = combined.evaluate("anything", CONFIG)
        assert assertion_result.passed is False
