"""AssertionRunner: orchestrates execution of assertions against a provider.

The runner handles: calling the provider, threading the response through
each assertion in order, respecting fail-fast configuration, collecting
results, and timing the entire operation. Retry logic for provider errors
lives here, not in individual assertions.

Supports both content assertions (on response text) and trajectory
assertions (on tool call sequences). When a case has trajectory
assertions, the runner extracts tool calls from the provider response
and evaluates trajectory assertions against a ToolCallTrajectory.
"""

from __future__ import annotations

import logging
import time

from callspec.assertions.base import BaseAssertion
from callspec.assertions.trajectory_base import TrajectoryAssertion
from callspec.core.config import CallspecConfig
from callspec.core.suite import AssertionSuite
from callspec.core.trajectory import ToolCall, ToolCallTrajectory
from callspec.core.types import (
    AssertionResult,
    IndividualAssertionResult,
    ProviderResponse,
    SuiteResult,
)
from callspec.errors import ProviderError
from callspec.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class AssertionRunner:
    """Executes assertion chains and suites against a configured provider.

    The runner is the bridge between the provider layer and the assertion layer.
    Assertions are pure functions over strings; the runner supplies the string
    by calling the provider, then collects assertion results into a structured
    report.
    """

    def __init__(self, provider: BaseProvider, config: CallspecConfig | None = None) -> None:
        self._provider = provider
        self._config = config or CallspecConfig()

    @property
    def config(self) -> CallspecConfig:
        return self._config

    @property
    def provider(self) -> BaseProvider:
        return self._provider

    def run_assertions(
        self,
        prompt: str,
        assertions: list[BaseAssertion],
        messages: list[dict[str, str]] | None = None,
    ) -> AssertionResult:
        """Run a list of assertions against a single prompt.

        Calls the provider once, then evaluates each assertion against the
        response content. Respects fail_fast: if True, stops at first failure.
        """
        start_time = time.monotonic()
        logger.debug("Running %d assertion(s) against prompt (%d chars)", len(assertions), len(prompt))

        provider_response = self._call_provider_with_retries(prompt, messages)
        content = provider_response.content
        logger.debug("Provider returned %d chars from model %s", len(content), provider_response.model)

        individual_results: list[IndividualAssertionResult] = []
        all_passed = True

        for assertion in assertions:
            logger.debug("Evaluating %s.%s", assertion.assertion_type, assertion.assertion_name)
            individual = assertion.evaluate(content, self._config)
            individual_results.append(individual)

            if not individual.passed:
                all_passed = False
                logger.debug(
                    "Assertion %s.%s FAILED: %s",
                    assertion.assertion_type,
                    assertion.assertion_name,
                    individual.message,
                )
                if self._config.fail_fast:
                    logger.debug("Fail-fast enabled, stopping assertion chain")
                    break
            else:
                logger.debug("Assertion %s.%s passed", assertion.assertion_type, assertion.assertion_name)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        logger.debug("Assertions complete in %dms, all_passed=%s", elapsed_ms, all_passed)

        return AssertionResult(
            passed=all_passed,
            assertions=individual_results,
            provider_response=provider_response,
            execution_time_ms=elapsed_ms,
            model=provider_response.model,
            prompt_tokens=provider_response.prompt_tokens,
            completion_tokens=provider_response.completion_tokens,
        )

    def run_suite(self, suite: AssertionSuite) -> SuiteResult:
        """Run a full assertion suite: multiple cases, each with their own assertions.

        Cases can contain content assertions, trajectory assertions, or both.
        The runner calls the provider once per case and evaluates all applicable
        assertion types against the response.
        """
        start_time = time.monotonic()
        logger.debug("Running suite with %d case(s)", len(suite.cases))
        case_results: dict[str, AssertionResult] = {}
        passed_count = 0
        failed_count = 0

        for case in suite.cases:
            logger.debug("Starting case '%s'", case.name)
            case_result = self._run_case(case.prompt, case.messages, case)
            case_results[case.name] = case_result

            if case_result.passed:
                passed_count += 1
                logger.debug("Case '%s' passed", case.name)
            else:
                failed_count += 1
                logger.debug("Case '%s' FAILED", case.name)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        all_passed = failed_count == 0
        logger.debug(
            "Suite complete in %dms: %d passed, %d failed",
            elapsed_ms, passed_count, failed_count,
        )

        return SuiteResult(
            passed=all_passed,
            case_results=case_results,
            total_cases=len(suite.cases),
            passed_cases=passed_count,
            failed_cases=failed_count,
            warned_cases=0,
            execution_time_ms=elapsed_ms,
        )

    def _run_case(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None,
        case,
    ) -> AssertionResult:
        """Run a single case: call provider, evaluate content + trajectory assertions."""
        start_time = time.monotonic()
        provider_response = self._call_provider_with_retries(prompt, messages)

        individual_results: list[IndividualAssertionResult] = []
        all_passed = True

        # Content assertions evaluate against response text
        if case.has_content_assertions:
            content = provider_response.content
            for assertion in case.assertions:
                logger.debug("Evaluating %s.%s", assertion.assertion_type, assertion.assertion_name)
                individual = assertion.evaluate(content, self._config)
                individual_results.append(individual)
                if not individual.passed:
                    all_passed = False
                    logger.debug(
                        "Assertion %s.%s FAILED: %s",
                        assertion.assertion_type, assertion.assertion_name, individual.message,
                    )
                    if self._config.fail_fast:
                        break
                else:
                    logger.debug("Assertion %s.%s passed", assertion.assertion_type, assertion.assertion_name)

        # Trajectory assertions evaluate against extracted tool calls
        if case.has_trajectory_assertions and (all_passed or not self._config.fail_fast):
            trajectory = self._response_to_trajectory(provider_response)
            logger.debug(
                "Evaluating %d trajectory assertion(s) against %d tool call(s)",
                len(case.trajectory_assertions), len(trajectory),
            )
            for assertion in case.trajectory_assertions:
                logger.debug("Evaluating %s.%s", assertion.assertion_type, assertion.assertion_name)
                individual = assertion.evaluate_trajectory(trajectory, self._config)
                individual_results.append(individual)
                if not individual.passed:
                    all_passed = False
                    logger.debug(
                        "Assertion %s.%s FAILED: %s",
                        assertion.assertion_type, assertion.assertion_name, individual.message,
                    )
                    if self._config.fail_fast:
                        break
                else:
                    logger.debug("Assertion %s.%s passed", assertion.assertion_type, assertion.assertion_name)

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        return AssertionResult(
            passed=all_passed,
            assertions=individual_results,
            provider_response=provider_response,
            execution_time_ms=elapsed_ms,
            model=provider_response.model,
            prompt_tokens=provider_response.prompt_tokens,
            completion_tokens=provider_response.completion_tokens,
        )

    @staticmethod
    def _response_to_trajectory(response: ProviderResponse) -> ToolCallTrajectory:
        """Extract a ToolCallTrajectory from a provider response."""
        calls = [
            ToolCall(
                tool_name=tc.get("name", tc.get("tool_name", "unknown")),
                arguments=tc.get("arguments", {}),
                call_index=i,
            )
            for i, tc in enumerate(response.tool_calls)
        ]
        return ToolCallTrajectory(
            calls=calls,
            model=response.model,
            provider=response.provider,
            raw_response=response.raw,
        )

    def _call_provider_with_retries(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> ProviderResponse:
        """Call the provider with exponential backoff on transient errors.

        Retries on provider/network errors only. Assertion failures are never retried
        because they indicate a behavioral problem, not an infrastructure problem.
        """
        last_error: Exception | None = None

        for attempt in range(1, self._config.max_retries + 1):
            try:
                logger.debug(
                    "Provider call attempt %d/%d to %s",
                    attempt, self._config.max_retries, self._provider.provider_name,
                )
                return self._provider.call(
                    prompt=prompt,
                    messages=messages,
                    temperature=self._config.temperature,
                    seed=self._config.seed,
                )
            except Exception as provider_error:
                last_error = provider_error
                if attempt < self._config.max_retries:
                    backoff_seconds = (
                        self._config.retry_backoff_base_seconds * (2 ** (attempt - 1))
                    )
                    logger.warning(
                        "Provider call attempt %d failed (%s: %s), retrying in %.1fs",
                        attempt, type(provider_error).__name__, provider_error, backoff_seconds,
                    )
                    time.sleep(backoff_seconds)
                else:
                    logger.error(
                        "Provider call failed after %d attempt(s): %s",
                        attempt, provider_error,
                    )

        raise ProviderError(
            provider=self._provider.provider_name,
            message="Provider call failed after exhausting all retries.",
            attempts=self._config.max_retries,
            cause=last_error,
        )
