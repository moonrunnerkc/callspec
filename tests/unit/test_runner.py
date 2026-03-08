"""Unit tests for AssertionRunner, suite execution, and fail-fast behavior."""

from __future__ import annotations

import json

import pytest

from verdict.assertions.structural import ContainsKeys, IsValidJson, LengthBetween
from verdict.core.config import VerdictConfig
from verdict.core.runner import AssertionRunner
from verdict.core.suite import AssertionCase, AssertionSuite
from verdict.errors import ProviderError
from verdict.providers.mock import MockProvider


def _json_provider(content: dict) -> MockProvider:
    """Create a MockProvider that always returns the given dict as JSON."""
    serialized = json.dumps(content)
    return MockProvider(lambda prompt, messages: serialized)


class TestAssertionRunner:

    def test_single_passing_assertion(self) -> None:
        provider = _json_provider({"title": "Test"})
        runner = AssertionRunner(provider)
        assertion_result = runner.run_assertions(
            prompt="test",
            assertions=[IsValidJson()],
        )
        assert assertion_result.passed is True
        assert len(assertion_result.assertions) == 1
        assert assertion_result.model == "mock"

    def test_single_failing_assertion(self) -> None:
        provider = MockProvider(lambda prompt, messages: "not json")
        runner = AssertionRunner(provider)
        assertion_result = runner.run_assertions(
            prompt="test",
            assertions=[IsValidJson()],
        )
        assert assertion_result.passed is False

    def test_multiple_passing_assertions(self) -> None:
        provider = _json_provider({"title": "Test"})
        runner = AssertionRunner(provider)
        assertion_result = runner.run_assertions(
            prompt="test",
            assertions=[
                IsValidJson(),
                ContainsKeys(["title"]),
                LengthBetween(1, 1000),
            ],
        )
        assert assertion_result.passed is True
        assert len(assertion_result.assertions) == 3

    def test_fail_fast_stops_at_first_failure(self) -> None:
        provider = MockProvider(lambda prompt, messages: "short")
        config = VerdictConfig(fail_fast=True)
        runner = AssertionRunner(provider, config)

        assertion_result = runner.run_assertions(
            prompt="test",
            assertions=[
                IsValidJson(),  # fails
                LengthBetween(1, 5),  # would pass but should not run
            ],
        )
        assert assertion_result.passed is False
        # Only one assertion should have been evaluated due to fail_fast
        assert len(assertion_result.assertions) == 1

    def test_no_fail_fast_runs_all(self) -> None:
        provider = MockProvider(lambda prompt, messages: "short")
        config = VerdictConfig(fail_fast=False)
        runner = AssertionRunner(provider, config)

        assertion_result = runner.run_assertions(
            prompt="test",
            assertions=[
                IsValidJson(),  # fails
                LengthBetween(1, 100),  # passes
            ],
        )
        assert assertion_result.passed is False
        assert len(assertion_result.assertions) == 2
        assert assertion_result.assertions[0].passed is False
        assert assertion_result.assertions[1].passed is True

    def test_execution_time_recorded(self) -> None:
        provider = _json_provider({"key": "val"})
        runner = AssertionRunner(provider)
        assertion_result = runner.run_assertions("test", [IsValidJson()])
        assert assertion_result.execution_time_ms >= 0

    def test_provider_response_attached(self) -> None:
        provider = _json_provider({"result": True})
        runner = AssertionRunner(provider)
        assertion_result = runner.run_assertions("test", [IsValidJson()])
        assert assertion_result.provider_response is not None
        assert assertion_result.provider_response.provider == "mock"

    def test_retry_on_provider_error(self) -> None:
        call_count = 0

        def flaky_fn(prompt: str, messages: list) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network timeout")
            return '{"ok": true}'

        provider = MockProvider(flaky_fn)
        config = VerdictConfig(max_retries=3, retry_backoff_base_seconds=0.01)
        runner = AssertionRunner(provider, config)

        assertion_result = runner.run_assertions("test", [IsValidJson()])
        assert assertion_result.passed is True
        assert call_count == 3

    def test_exhausted_retries_raises(self) -> None:
        def always_fail(prompt: str, messages: list) -> str:
            raise ConnectionError("permanent failure")

        provider = MockProvider(always_fail)
        config = VerdictConfig(max_retries=2, retry_backoff_base_seconds=0.01)
        runner = AssertionRunner(provider, config)

        with pytest.raises(ProviderError, match="exhausting all retries"):
            runner.run_assertions("test", [IsValidJson()])


class TestSuiteExecution:

    def test_all_cases_pass(self) -> None:
        provider = _json_provider({"title": "Test", "body": "Content"})
        runner = AssertionRunner(provider)

        suite = AssertionSuite(name="test-suite")
        suite.add_case(AssertionCase(
            name="case-1",
            prompt="test prompt 1",
            assertions=[IsValidJson(), ContainsKeys(["title"])],
        ))
        suite.add_case(AssertionCase(
            name="case-2",
            prompt="test prompt 2",
            assertions=[LengthBetween(1, 1000)],
        ))

        suite_result = runner.run_suite(suite)
        assert suite_result.passed is True
        assert suite_result.total_cases == 2
        assert suite_result.passed_cases == 2
        assert suite_result.failed_cases == 0

    def test_one_case_fails(self) -> None:
        provider = MockProvider(lambda prompt, messages: "not json")
        runner = AssertionRunner(provider)

        suite = AssertionSuite(name="mixed-suite")
        suite.add_case(AssertionCase(
            name="passing-case",
            prompt="test",
            assertions=[LengthBetween(1, 100)],
        ))
        suite.add_case(AssertionCase(
            name="failing-case",
            prompt="test",
            assertions=[IsValidJson()],
        ))

        suite_result = runner.run_suite(suite)
        assert suite_result.passed is False
        assert suite_result.passed_cases == 1
        assert suite_result.failed_cases == 1

    def test_empty_suite(self) -> None:
        provider = _json_provider({})
        runner = AssertionRunner(provider)
        suite = AssertionSuite(name="empty")
        suite_result = runner.run_suite(suite)
        assert suite_result.passed is True
        assert suite_result.total_cases == 0
