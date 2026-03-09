"""AssertionSuite: an ordered collection of assertion cases with config.

A suite groups multiple test cases, each with a prompt and a list of
assertions. Suites can be constructed programmatically or parsed from
YAML configuration files. Both paths produce the same internal structure.

Cases come in two flavors:
1. Content cases: assertions on the text content of a provider response
2. Trajectory cases: assertions on the tool call sequence and arguments
   (set trajectory_assertions on the case, leave content assertions empty)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from callspec.assertions.base import BaseAssertion
from callspec.assertions.trajectory_base import TrajectoryAssertion
from callspec.core.config import CallspecConfig
from callspec.core.types import Severity


@dataclass
class AssertionCase:
    """A single test case within a suite: one prompt, multiple assertions.

    For content-based cases: populate `assertions` with BaseAssertion instances.
    For trajectory-based cases: populate `trajectory_assertions` with
    TrajectoryAssertion instances. A case can have both.
    """

    name: str
    prompt: str
    assertions: list[BaseAssertion] = field(default_factory=list)
    trajectory_assertions: list[TrajectoryAssertion] = field(default_factory=list)
    messages: list[dict[str, str]] | None = None
    severity: Severity = Severity.ERROR

    @property
    def has_trajectory_assertions(self) -> bool:
        return len(self.trajectory_assertions) > 0

    @property
    def has_content_assertions(self) -> bool:
        return len(self.assertions) > 0


@dataclass
class AssertionSuite:
    """Ordered collection of assertion cases plus suite-level configuration.

    The runner iterates cases in order. Fail-fast behavior (controlled by
    config) determines whether execution stops at the first failing case
    or runs all cases and reports all failures.
    """

    name: str
    cases: list[AssertionCase] = field(default_factory=list)
    config: CallspecConfig = field(default_factory=CallspecConfig)

    def add_case(self, case: AssertionCase) -> None:
        self.cases.append(case)

    @property
    def case_count(self) -> int:
        return len(self.cases)
