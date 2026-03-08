"""End-to-end tests for the Verdict class and fluent assertion API.

These verify the Phase 1 milestone: Verdict(MockProvider(...)).assert_that("test")
chains through structural assertions and produces correct AssertionResult objects.
"""

from __future__ import annotations

import json

from verdict import Verdict, VerdictConfig
from verdict.providers.mock import MockProvider


def _static_provider(content: str) -> MockProvider:
    """MockProvider that returns a fixed string regardless of input."""
    return MockProvider(lambda prompt, messages: content)


class TestVerdictEndToEnd:

    def test_basic_json_assertion(self) -> None:
        provider = _static_provider('{"title": "Hello"}')
        v = Verdict(provider)
        assertion_result = v.assert_that("test prompt").is_valid_json().run()
        assert assertion_result.passed is True
        assert assertion_result.model == "mock"

    def test_chained_structural_assertions(self) -> None:
        content = json.dumps({
            "title": "Summary Report",
            "summary": "This is a detailed summary of the document.",
            "word_count": 150,
        })
        provider = _static_provider(content)
        v = Verdict(provider)

        assertion_result = (
            v.assert_that("Summarize this document")
            .is_valid_json()
            .contains_keys(["title", "summary"])
            .length_between(10, 500)
            .run()
        )

        assert assertion_result.passed is True
        assert len(assertion_result.assertions) == 3
        assert all(a.passed for a in assertion_result.assertions)

    def test_chain_fails_on_missing_key(self) -> None:
        content = json.dumps({"title": "No summary here"})
        provider = _static_provider(content)
        v = Verdict(provider)

        assertion_result = (
            v.assert_that("test")
            .is_valid_json()
            .contains_keys(["title", "summary"])
            .run()
        )

        assert assertion_result.passed is False
        # First assertion (is_valid_json) passed, second (contains_keys) failed
        assert assertion_result.assertions[0].passed is True
        assert assertion_result.assertions[1].passed is False

    def test_schema_validation(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        }
        content = json.dumps({"answer": "42", "confidence": 0.95})
        provider = _static_provider(content)
        v = Verdict(provider)

        assertion_result = (
            v.assert_that("What is the answer?")
            .matches_schema(schema)
            .run()
        )
        assert assertion_result.passed is True

    def test_does_not_contain(self) -> None:
        provider = _static_provider("Our product is the best choice.")
        v = Verdict(provider)

        assertion_result = (
            v.assert_that("Recommend our product")
            .does_not_contain("competitor")
            .run()
        )
        assert assertion_result.passed is True

    def test_pattern_matching(self) -> None:
        provider = _static_provider("Version: 2.3.1")
        v = Verdict(provider)

        assertion_result = (
            v.assert_that("Show version")
            .matches_pattern(r"Version: \d+\.\d+\.\d+")
            .run()
        )
        assert assertion_result.passed is True

    def test_starts_and_ends_with(self) -> None:
        provider = _static_provider("BEGIN: some content :END")
        v = Verdict(provider)

        assertion_result = (
            v.assert_that("Format response")
            .starts_with("BEGIN:")
            .ends_with(":END")
            .run()
        )
        assert assertion_result.passed is True

    def test_custom_config(self) -> None:
        config = VerdictConfig(fail_fast=False)
        provider = MockProvider(lambda prompt, messages: "not json")
        v = Verdict(provider, config=config)

        assertion_result = (
            v.assert_that("test")
            .is_valid_json()
            .length_between(1, 100)
            .run()
        )

        # Both assertions run because fail_fast=False
        assert assertion_result.passed is False
        assert len(assertion_result.assertions) == 2

    def test_provider_response_accessible(self) -> None:
        provider = _static_provider("test content")
        v = Verdict(provider)
        assertion_result = v.assert_that("prompt").length_between(1, 100).run()
        assert assertion_result.provider_response is not None
        assert assertion_result.provider_response.content == "test content"

    def test_assertion_count_on_builder(self) -> None:
        provider = _static_provider("{}")
        v = Verdict(provider)
        builder = v.assert_that("test").is_valid_json().length_between(0, 100)
        assert builder.assertion_count == 2

    def test_not_assertion_via_builder(self) -> None:
        from verdict.assertions.structural import IsValidJson

        provider = _static_provider("plain text, not json")
        v = Verdict(provider)

        assertion_result = (
            v.assert_that("test")
            .not_(IsValidJson())
            .run()
        )
        assert assertion_result.passed is True

    def test_or_assertion_via_builder(self) -> None:
        from verdict.assertions.structural import StartsWith

        provider = _static_provider("Hello world")
        v = Verdict(provider)

        assertion_result = (
            v.assert_that("test")
            .or_(StartsWith("Hello"), StartsWith("Goodbye"))
            .run()
        )
        assert assertion_result.passed is True
