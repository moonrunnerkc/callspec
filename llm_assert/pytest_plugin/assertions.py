"""Pytest assertion helpers that produce Verdict-flavored failure output.

These functions bridge Verdict's AssertionResult into pytest's assertion
introspection. When a Verdict assertion fails, the developer sees:
which assertion type failed, the score vs threshold, the model and
provider, and the input prompt. Not just "assert False".

Usage in a test:
    from verdict.pytest_plugin.assertions import assert_verdict_pass

    def test_json_output(verdict_runner):
        result = verdict_runner.assert_that("Return JSON").is_valid_json().run()
        assert_verdict_pass(result)
"""

from __future__ import annotations

from verdict.core.types import AssertionResult, IndividualAssertionResult


class VerdictAssertionError(AssertionError):
    """Custom AssertionError with structured Verdict failure details.

    Inherits from AssertionError so pytest's native assertion
    introspection displays the message correctly.
    """

    def __init__(self, message: str, result: AssertionResult) -> None:
        self.verdict_result = result
        super().__init__(message)


def _format_individual_failure(individual: IndividualAssertionResult) -> str:
    """Format a single assertion failure for terminal output."""
    parts = [
        f"  [{individual.assertion_type}] {individual.assertion_name}: FAILED",
    ]

    if individual.score is not None:
        parts.append(f"    score: {individual.score:.4f}")
    if individual.threshold is not None:
        parts.append(f"    threshold: {individual.threshold:.4f}")
    if individual.confidence is not None:
        parts.append(f"    confidence: {individual.confidence:.4f}")

    parts.append(f"    {individual.message}")

    if individual.details:
        for key, value in individual.details.items():
            # Skip large nested structures in the summary line
            if isinstance(value, (list, dict)) and len(str(value)) > 200:
                parts.append(f"    {key}: ({type(value).__name__}, {len(value)} items)")
            else:
                parts.append(f"    {key}: {value}")

    return "\n".join(parts)


def _format_failure_report(result: AssertionResult) -> str:
    """Build the full failure report string shown in pytest output."""
    lines = ["", "Verdict assertion chain FAILED", ""]

    # Provider and model context
    if result.provider_response:
        lines.append(
            f"  provider: {result.provider_response.provider}, "
            f"model: {result.provider_response.model}"
        )
    lines.append(f"  execution time: {result.execution_time_ms}ms")

    if result.prompt_tokens is not None:
        lines.append(
            f"  tokens: {result.prompt_tokens} prompt, "
            f"{result.completion_tokens} completion"
        )

    lines.append("")

    # Individual assertion results
    for individual in result.assertions:
        if individual.passed:
            lines.append(f"  [{individual.assertion_type}] {individual.assertion_name}: PASSED")
        else:
            lines.append(_format_individual_failure(individual))

    lines.append("")
    return "\n".join(lines)


def assert_verdict_pass(result: AssertionResult) -> None:
    """Assert that a Verdict result passed, with structured failure output.

    Use this instead of `assert result.passed` to get Verdict-specific
    failure formatting in the pytest terminal output.
    """
    if not result.passed:
        report = _format_failure_report(result)
        raise VerdictAssertionError(report, result)


def assert_verdict_fail(result: AssertionResult) -> None:
    """Assert that a Verdict result failed (for testing negative assertions).

    Useful when writing tests that verify a model correctly rejects input.
    """
    if result.passed:
        raise AssertionError(
            "Expected Verdict assertion chain to FAIL, but all assertions passed.\n"
            f"  assertions: {len(result.assertions)}\n"
            f"  model: {result.model}\n"
        )
