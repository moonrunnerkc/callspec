"""Tests for Rich report rendering: render_rich_report and _render_rich_case.

Uses a captured Rich Console (file=StringIO) to verify the rendered output
includes the expected structural elements and content without asserting
on exact formatting, which is fragile across Rich versions.
"""

from __future__ import annotations

import io

from rich.console import Console

from callspec.cli.console import CALLSPEC_THEME
from callspec.core.report import _render_rich_case, render_rich_report
from callspec.core.types import (
    AssertionResult,
    IndividualAssertionResult,
    ProviderResponse,
    SuiteResult,
)


def _make_provider_response(content: str = "test response") -> ProviderResponse:
    return ProviderResponse(
        content=content,
        raw={},
        model="test-model-v1",
        provider="mock",
        latency_ms=42,
        prompt_tokens=10,
        completion_tokens=20,
        finish_reason="stop",
        request_id=None,
        tool_calls=[],
    )


def _make_passing_result() -> AssertionResult:
    return AssertionResult(
        passed=True,
        assertions=[
            IndividualAssertionResult(
                assertion_type="structural",
                assertion_name="is_valid_json",
                passed=True,
                message="Content is valid JSON.",
            ),
        ],
        provider_response=_make_provider_response(),
        execution_time_ms=15,
        model="test-model-v1",
        prompt_tokens=10,
        completion_tokens=20,
    )


def _make_failing_result() -> AssertionResult:
    return AssertionResult(
        passed=False,
        assertions=[
            IndividualAssertionResult(
                assertion_type="structural",
                assertion_name="is_valid_json",
                passed=True,
                message="Content is valid JSON.",
            ),
            IndividualAssertionResult(
                assertion_type="structural",
                assertion_name="contains_keys",
                passed=False,
                message="Missing keys: ['summary']. Found: ['title'].",
                score=0.5,
                threshold=1.0,
            ),
        ],
        provider_response=_make_provider_response(),
        execution_time_ms=23,
        model="test-model-v1",
        prompt_tokens=10,
        completion_tokens=20,
    )


def _make_suite_result(passed: bool) -> SuiteResult:
    case_results = {
        "valid_json_case": _make_passing_result(),
    }
    if not passed:
        case_results["missing_keys_case"] = _make_failing_result()

    return SuiteResult(
        passed=passed,
        case_results=case_results,
        total_cases=len(case_results),
        passed_cases=1,
        failed_cases=0 if passed else 1,
        execution_time_ms=38 if not passed else 15,
    )


def _capture_rich_output(fn, *args, **kwargs) -> str:
    """Temporarily redirect the callspec console to a StringIO buffer.

    Patches callspec.cli.console.console and callspec.core.report's
    reference to it, captures output, then restores the original.
    """
    buf = io.StringIO()
    capture_console = Console(
        file=buf,
        theme=CALLSPEC_THEME,
        force_terminal=True,
        width=120,
    )

    import callspec.cli.console as console_mod

    original_console = console_mod.console
    console_mod.console = capture_console

    try:
        fn(*args, **kwargs)
    finally:
        console_mod.console = original_console

    return buf.getvalue()


class TestRenderRichReport:
    """Tests for the top-level render_rich_report function."""

    def test_passing_suite_shows_passed_status(self) -> None:
        suite_result = _make_suite_result(passed=True)
        output = _capture_rich_output(
            render_rich_report, suite_result=suite_result, suite_name="Test Suite"
        )
        assert "PASSED" in output
        assert "Test Suite" in output

    def test_failing_suite_shows_failed_status(self) -> None:
        suite_result = _make_suite_result(passed=False)
        output = _capture_rich_output(
            render_rich_report, suite_result=suite_result, suite_name="Failing Suite"
        )
        assert "FAILED" in output
        assert "Failing Suite" in output

    def test_suite_shows_case_count(self) -> None:
        suite_result = _make_suite_result(passed=True)
        output = _capture_rich_output(
            render_rich_report, suite_result=suite_result, suite_name="Count Suite"
        )
        assert "1/1 passed" in output

    def test_suite_shows_duration(self) -> None:
        suite_result = _make_suite_result(passed=True)
        output = _capture_rich_output(
            render_rich_report, suite_result=suite_result, suite_name="Timing Suite"
        )
        assert "15ms" in output

    def test_suite_renders_each_case(self) -> None:
        suite_result = _make_suite_result(passed=False)
        output = _capture_rich_output(
            render_rich_report, suite_result=suite_result, suite_name="Multi"
        )
        assert "valid_json_case" in output
        assert "missing_keys_case" in output

    def test_assertion_results_without_suite(self) -> None:
        """render_rich_report also accepts bare assertion_results dict."""
        output = _capture_rich_output(
            render_rich_report,
            assertion_results={"standalone_case": _make_passing_result()},
        )
        assert "standalone_case" in output
        assert "is_valid_json" in output


class TestRenderRichCase:
    """Tests for the _render_rich_case helper."""

    def test_passing_case_shows_check_mark(self) -> None:
        output = _capture_rich_output(
            _render_rich_case, "my_case", _make_passing_result()
        )
        # Unicode check mark
        assert "\u2713" in output

    def test_failing_case_shows_cross_mark(self) -> None:
        output = _capture_rich_output(
            _render_rich_case, "fail_case", _make_failing_result()
        )
        # Unicode cross mark
        assert "\u2717" in output

    def test_case_shows_model_and_timing(self) -> None:
        output = _capture_rich_output(
            _render_rich_case, "timed_case", _make_passing_result()
        )
        assert "test-model-v1" in output
        assert "15ms" in output

    def test_case_shows_assertion_names(self) -> None:
        output = _capture_rich_output(
            _render_rich_case, "detailed_case", _make_failing_result()
        )
        assert "is_valid_json" in output
        assert "contains_keys" in output

    def test_failing_assertion_shows_message(self) -> None:
        output = _capture_rich_output(
            _render_rich_case, "msg_case", _make_failing_result()
        )
        assert "Missing keys" in output

    def test_score_displayed_for_scored_assertions(self) -> None:
        output = _capture_rich_output(
            _render_rich_case, "score_case", _make_failing_result()
        )
        assert "0.5000" in output
        assert "1.0000" in output
