"""AssertionRunner: orchestrates execution of assertions against a provider.

The runner handles: calling the provider, threading the response through
each assertion in order, respecting fail-fast configuration, collecting
results, and timing the entire operation. Retry logic for provider errors
lives here, not in individual assertions.
"""

from __future__ import annotations

import time

from verdict.assertions.base import BaseAssertion
from verdict.core.config import VerdictConfig
from verdict.core.suite import AssertionSuite
from verdict.core.types import (
    AssertionResult,
    IndividualAssertionResult,
    ProviderResponse,
    SuiteResult,
)
from verdict.errors import ProviderError
from verdict.providers.base import BaseProvider


class AssertionRunner:
    """Executes assertion chains and suites against a configured provider.

    The runner is the bridge between the provider layer and the assertion layer.
    Assertions are pure functions over strings; the runner supplies the string
    by calling the provider, then collects assertion results into a structured
    report.
    """

    def __init__(self, provider: BaseProvider, config: VerdictConfig | None = None) -> None:
        self._provider = provider
        self._config = config or VerdictConfig()

    @property
    def config(self) -> VerdictConfig:
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

        provider_response = self._call_provider_with_retries(prompt, messages)
        content = provider_response.content

        individual_results: list[IndividualAssertionResult] = []
        all_passed = True

        for assertion in assertions:
            individual = assertion.evaluate(content, self._config)
            individual_results.append(individual)

            if not individual.passed:
                all_passed = False
                if self._config.fail_fast:
                    break

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

    def run_suite(self, suite: AssertionSuite) -> SuiteResult:
        """Run a full assertion suite: multiple cases, each with their own assertions."""
        start_time = time.monotonic()
        case_results: dict[str, AssertionResult] = {}
        passed_count = 0
        failed_count = 0

        # Use suite-level config if provided, falling back to runner config

        for case in suite.cases:
            case_result = self.run_assertions(
                prompt=case.prompt,
                assertions=case.assertions,
                messages=case.messages,
            )
            case_results[case.name] = case_result

            if case_result.passed:
                passed_count += 1
            else:
                failed_count += 1

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        all_passed = failed_count == 0

        return SuiteResult(
            passed=all_passed,
            case_results=case_results,
            total_cases=len(suite.cases),
            passed_cases=passed_count,
            failed_cases=failed_count,
            warned_cases=0,
            execution_time_ms=elapsed_ms,
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
                    time.sleep(backoff_seconds)

        raise ProviderError(
            provider=self._provider.provider_name,
            message="Provider call failed after exhausting all retries.",
            attempts=self._config.max_retries,
            cause=last_error,
        )
