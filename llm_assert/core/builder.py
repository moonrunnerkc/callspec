"""AssertionBuilder: fluent API for chaining assertions.

This is the developer-facing API. A call to llm_assert.assert_that(prompt)
returns an AssertionBuilder. The developer chains structural, regression,
and composite assertions, then calls .run() to execute.

The provider call is deferred until .run(): building the chain is free,
which allows assertion chains to be constructed in setup code and
executed conditionally.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from llm_assert.assertions.base import BaseAssertion
from llm_assert.assertions.composite import NegationWrapper, OrAssertion
from llm_assert.assertions.regression import (
    FormatMatchesBaseline,
    MatchesBaseline,
)
from llm_assert.assertions.structural import (
    ContainsKeys,
    DoesNotContain,
    EndsWith,
    IsValidJson,
    LengthBetween,
    MatchesPattern,
    MatchesSchema,
    StartsWith,
)
from llm_assert.core.runner import AssertionRunner
from llm_assert.core.types import AssertionResult
from llm_assert.snapshots.manager import SnapshotManager


class AssertionBuilder:
    """Fluent builder for assertion chains.

    Each method appends an assertion to the chain and returns self,
    enabling natural chaining syntax. The .run() method triggers
    the provider call and assertion evaluation.
    """

    def __init__(
        self,
        runner: AssertionRunner,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        self._runner = runner
        self._prompt = prompt
        self._messages = messages
        self._assertions: list[BaseAssertion] = []

    # -- Structural assertions --

    def is_valid_json(self) -> AssertionBuilder:
        """Assert the response is valid JSON."""
        self._assertions.append(IsValidJson())
        return self

    def matches_schema(self, schema: dict[str, Any]) -> AssertionBuilder:
        """Assert the response validates against a JSON Schema."""
        self._assertions.append(MatchesSchema(schema))
        return self

    def contains_keys(self, keys: Sequence[str]) -> AssertionBuilder:
        """Assert the response JSON contains all specified top-level keys."""
        self._assertions.append(ContainsKeys(keys))
        return self

    def length_between(self, min_chars: int, max_chars: int) -> AssertionBuilder:
        """Assert the response length falls within [min_chars, max_chars]."""
        self._assertions.append(LengthBetween(min_chars, max_chars))
        return self

    def matches_pattern(self, pattern: str) -> AssertionBuilder:
        """Assert the response matches a regular expression."""
        self._assertions.append(MatchesPattern(pattern))
        return self

    def does_not_contain(
        self, text_or_pattern: str, is_regex: bool = False
    ) -> AssertionBuilder:
        """Assert the response does not contain the specified text or pattern."""
        self._assertions.append(DoesNotContain(text_or_pattern, is_regex))
        return self

    def starts_with(self, prefix: str) -> AssertionBuilder:
        """Assert the response starts with the specified prefix."""
        self._assertions.append(StartsWith(prefix))
        return self

    def ends_with(self, suffix: str) -> AssertionBuilder:
        """Assert the response ends with the specified suffix."""
        self._assertions.append(EndsWith(suffix))
        return self

    # -- Regression assertions --

    def matches_baseline(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
    ) -> AssertionBuilder:
        """Assert the response matches a recorded baseline structurally.

        Compares JSON structure (top-level keys) and content. Loads
        baseline from the provided SnapshotManager.
        """
        self._assertions.append(
            MatchesBaseline(snapshot_key, snapshot_manager)
        )
        return self

    def format_matches_baseline(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
    ) -> AssertionBuilder:
        """Assert the response JSON structure matches the baseline format.

        Structural-only comparison. Does not evaluate semantic similarity.
        Useful when the model produces the same format but different content.
        """
        self._assertions.append(
            FormatMatchesBaseline(snapshot_key, snapshot_manager)
        )
        return self

    # -- Composite assertions --

    def not_(self, assertion: BaseAssertion) -> AssertionBuilder:
        """Negate an assertion: passes when the inner assertion fails."""
        self._assertions.append(NegationWrapper(assertion))
        return self

    def or_(self, *assertions: BaseAssertion) -> AssertionBuilder:
        """Pass if at least one of the given assertions passes."""
        self._assertions.append(OrAssertion(list(assertions)))
        return self

    # -- Custom assertion --

    def satisfies(self, assertion: BaseAssertion) -> AssertionBuilder:
        """Add a custom assertion implementing BaseAssertion."""
        self._assertions.append(assertion)
        return self

    # -- Execution --

    def run(self) -> AssertionResult:
        """Execute the provider call and evaluate all chained assertions.

        Returns an AssertionResult with pass/fail, individual results,
        provider response, and execution metadata.
        """
        return self._runner.run_assertions(
            prompt=self._prompt,
            assertions=self._assertions,
            messages=self._messages,
        )

    @property
    def assertion_count(self) -> int:
        """Number of assertions in the current chain."""
        return len(self._assertions)
