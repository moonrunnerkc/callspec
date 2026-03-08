"""AssertionSuite: an ordered collection of assertion cases with config.

A suite groups multiple test cases, each with a prompt and a list of
assertions. Suites can be constructed programmatically or parsed from
YAML configuration files. Both paths produce the same internal structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_assert.assertions.base import BaseAssertion
from llm_assert.core.config import LLMAssertConfig
from llm_assert.core.types import Severity


@dataclass
class AssertionCase:
    """A single test case within a suite: one prompt, multiple assertions."""

    name: str
    prompt: str
    assertions: list[BaseAssertion]
    messages: list[dict[str, str]] | None = None
    severity: Severity = Severity.ERROR


@dataclass
class AssertionSuite:
    """Ordered collection of assertion cases plus suite-level configuration.

    The runner iterates cases in order. Fail-fast behavior (controlled by
    config) determines whether execution stops at the first failing case
    or runs all cases and reports all failures.
    """

    name: str
    cases: list[AssertionCase] = field(default_factory=list)
    config: LLMAssertConfig = field(default_factory=LLMAssertConfig)

    def add_case(self, case: AssertionCase) -> None:
        self.cases.append(case)

    @property
    def case_count(self) -> int:
        return len(self.cases)
