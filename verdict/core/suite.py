"""AssertionSuite: an ordered collection of assertion cases with config.

A suite groups multiple test cases, each with a prompt and a list of
assertions. Suites can be constructed programmatically or parsed from
YAML configuration files. Both paths produce the same internal structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from verdict.assertions.base import BaseAssertion
from verdict.core.config import VerdictConfig
from verdict.core.types import Severity


@dataclass
class AssertionCase:
    """A single test case within a suite: one prompt, multiple assertions."""

    name: str
    prompt: str
    assertions: List[BaseAssertion]
    messages: Optional[List[Dict[str, str]]] = None
    severity: Severity = Severity.ERROR


@dataclass
class AssertionSuite:
    """Ordered collection of assertion cases plus suite-level configuration.

    The runner iterates cases in order. Fail-fast behavior (controlled by
    config) determines whether execution stops at the first failing case
    or runs all cases and reports all failures.
    """

    name: str
    cases: List[AssertionCase] = field(default_factory=list)
    config: VerdictConfig = field(default_factory=VerdictConfig)

    def add_case(self, case: AssertionCase) -> None:
        self.cases.append(case)

    @property
    def case_count(self) -> int:
        return len(self.cases)
