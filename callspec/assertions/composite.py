"""Negation and composite assertion wrappers.

NegationWrapper inverts any assertion. AndAssertion and OrAssertion compose
assertions with boolean logic. These are the building blocks for complex
behavioral contracts expressed as single chains.
"""

from __future__ import annotations

from callspec.assertions.base import BaseAssertion
from callspec.core.config import CallspecConfig
from callspec.core.types import IndividualAssertionResult


class NegationWrapper(BaseAssertion):
    """Inverts the result of any wrapped assertion."""

    assertion_type = "composite"
    assertion_name = "not"

    def __init__(self, inner: BaseAssertion) -> None:
        self._inner = inner
        self.assertion_name = f"not_{inner.assertion_name}"

    def evaluate(self, content: str, config: CallspecConfig) -> IndividualAssertionResult:
        inner_result = self._inner.evaluate(content, config)

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=not inner_result.passed,
            message=(
                f"Negation of '{self._inner.assertion_name}': "
                f"inner {'passed' if inner_result.passed else 'failed'}, "
                f"so negation {'failed' if inner_result.passed else 'passed'}."
            ),
            score=inner_result.score,
            threshold=inner_result.threshold,
            details={"inner_result": inner_result.message},
        )


class AndAssertion(BaseAssertion):
    """Passes only if all inner assertions pass."""

    assertion_type = "composite"
    assertion_name = "and"

    def __init__(self, assertions: list[BaseAssertion]) -> None:
        self._assertions = assertions

    def evaluate(self, content: str, config: CallspecConfig) -> IndividualAssertionResult:
        failures: list[str] = []
        all_details: list[dict] = []

        for assertion in self._assertions:
            individual = assertion.evaluate(content, config)
            all_details.append({
                "name": assertion.assertion_name,
                "passed": individual.passed,
                "message": individual.message,
            })
            if not individual.passed:
                failures.append(f"{assertion.assertion_name}: {individual.message}")

        passed = len(failures) == 0

        if passed:
            message = f"All {len(self._assertions)} assertions passed."
        else:
            message = (
                f"{len(failures)} of {len(self._assertions)} assertions failed. "
                f"First failure: {failures[0]}"
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            details={"assertion_results": all_details},
        )


class OrAssertion(BaseAssertion):
    """Passes if at least one inner assertion passes."""

    assertion_type = "composite"
    assertion_name = "or"

    def __init__(self, assertions: list[BaseAssertion]) -> None:
        self._assertions = assertions

    def evaluate(self, content: str, config: CallspecConfig) -> IndividualAssertionResult:
        all_details: list[dict] = []
        any_passed = False

        for assertion in self._assertions:
            individual = assertion.evaluate(content, config)
            all_details.append({
                "name": assertion.assertion_name,
                "passed": individual.passed,
                "message": individual.message,
            })
            if individual.passed:
                any_passed = True

        if any_passed:
            message = (
                f"At least one of {len(self._assertions)} assertions passed."
            )
        else:
            message = (
                f"None of {len(self._assertions)} assertions passed."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=any_passed,
            message=message,
            details={"assertion_results": all_details},
        )
