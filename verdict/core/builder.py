"""AssertionBuilder: fluent API for chaining assertions.

This is the developer-facing API. A call to verdict.assert_that(prompt)
returns an AssertionBuilder. The developer chains structural, semantic,
behavioral, and composite assertions, then calls .run() to execute.

The provider call is deferred until .run(): building the chain is free,
which allows assertion chains to be constructed in setup code and
executed conditionally.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from verdict.assertions.base import BaseAssertion
from verdict.assertions.behavioral import (
    IsConsistentAcrossSamples,
    PassesRate,
    RefusalRateIsAbove,
)
from verdict.assertions.composite import NegationWrapper, OrAssertion
from verdict.assertions.regression import (
    FormatMatchesBaseline,
    MatchesBaseline,
    SemanticDriftIsBelow,
)
from verdict.assertions.semantic import (
    DoesNotDiscuss,
    IsFactuallyConsistentWith,
    SemanticIntentMatches,
    UsesLanguageAtGradeLevel,
)
from verdict.assertions.structural import (
    ContainsKeys,
    DoesNotContain,
    EndsWith,
    IsValidJson,
    LengthBetween,
    MatchesPattern,
    MatchesSchema,
    StartsWith,
)
from verdict.core.runner import AssertionRunner
from verdict.core.types import AssertionResult
from verdict.sampling.sampler import BaseSampler
from verdict.sampling.seed import SeedManager
from verdict.snapshots.manager import SnapshotManager


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
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        self._runner = runner
        self._prompt = prompt
        self._messages = messages
        self._assertions: List[BaseAssertion] = []

    # -- Structural assertions --

    def is_valid_json(self) -> AssertionBuilder:
        """Assert the response is valid JSON."""
        self._assertions.append(IsValidJson())
        return self

    def matches_schema(self, schema: Dict[str, Any]) -> AssertionBuilder:
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

    # -- Semantic assertions --

    def semantic_intent_matches(
        self,
        reference_intent: str,
        threshold: float | None = None,
    ) -> AssertionBuilder:
        """Assert the response semantically aligns with the reference intent.

        Default threshold 0.75 calibrated against SBERT STS-B benchmarks.
        Requires verdict[semantic] extra.
        """
        self._assertions.append(SemanticIntentMatches(reference_intent, threshold))
        return self

    def does_not_discuss(
        self,
        topic: str,
        threshold: float | None = None,
    ) -> AssertionBuilder:
        """Assert the response does not discuss a prohibited topic.

        Uses a lower threshold (default 0.6) to catch loose topical proximity.
        """
        self._assertions.append(DoesNotDiscuss(topic, threshold))
        return self

    def is_factually_consistent_with(
        self,
        reference_text: str,
        threshold: float | None = None,
    ) -> AssertionBuilder:
        """Assert the response is semantically consistent with reference text.

        Consistency check, not factual accuracy. If the reference is wrong,
        a response repeating the wrong information will pass.
        """
        self._assertions.append(IsFactuallyConsistentWith(reference_text, threshold))
        return self

    def uses_language_at_grade_level(
        self,
        grade: int,
        tolerance: int = 2,
    ) -> AssertionBuilder:
        """Assert the response readability falls within a target grade range.

        Uses Flesch-Kincaid grade level formula. Zero API cost.
        """
        self._assertions.append(UsesLanguageAtGradeLevel(grade, tolerance))
        return self

    # -- Regression assertions --

    def matches_baseline(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
        semantic_threshold: float | None = None,
    ) -> AssertionBuilder:
        """Assert the response matches a recorded baseline structurally and semantically.

        Both checks must pass independently. Requires verdict[semantic] for
        the semantic comparison. Loads baseline from the provided SnapshotManager.
        """
        self._assertions.append(
            MatchesBaseline(snapshot_key, snapshot_manager, semantic_threshold)
        )
        return self

    def semantic_drift_is_below(
        self,
        snapshot_key: str,
        snapshot_manager: SnapshotManager,
        max_drift: float | None = None,
    ) -> AssertionBuilder:
        """Assert that semantic distance from baseline stays below max_drift.

        Drift is (1 - cosine_similarity). Tolerates structural changes.
        Default max_drift from config.regression_drift_ceiling (0.15).
        """
        self._assertions.append(
            SemanticDriftIsBelow(snapshot_key, snapshot_manager, max_drift)
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

    # -- Behavioral assertions --

    def passes_rate(
        self,
        assertion: BaseAssertion,
        min_rate: Optional[float] = None,
        n_samples: Optional[int] = None,
        sampler: Optional[BaseSampler] = None,
    ) -> AssertionBuilder:
        """Assert that an inner assertion passes at min_rate across N samples.

        Wraps any BaseAssertion into a behavioral check. The provider is
        called n_samples times (varied by sampler, or repeated prompt if
        no sampler). Uses Wilson confidence intervals for statistical rigor.

        Defaults from config: min_rate=0.95, n_samples=20.
        """
        seed = SeedManager(self._runner.config.seed)
        self._assertions.append(
            PassesRate(
                inner_assertion=assertion,
                provider=self._runner.provider,
                min_rate=min_rate,
                n_samples=n_samples,
                sampler=sampler,
                seed_manager=seed,
            )
        )
        return self

    def refusal_rate_is_above(
        self,
        threshold: Optional[float] = None,
        n_samples: Optional[int] = None,
        sampler: Optional[BaseSampler] = None,
        custom_patterns: Optional[Sequence[re.Pattern | str]] = None,
    ) -> AssertionBuilder:
        """Assert the model reliably refuses this class of input.

        Runs n_samples calls, classifies each response using the built-in
        refusal pattern library (OpenAI, Anthropic, Google phrasings) plus
        any custom_patterns provided. Passes if refusal rate meets threshold.

        Requires an InputSampler for meaningful results: testing refusal on
        a single repeated prompt measures consistency, not behavioral coverage.
        """
        seed = SeedManager(self._runner.config.seed)
        self._assertions.append(
            RefusalRateIsAbove(
                provider=self._runner.provider,
                threshold=threshold,
                n_samples=n_samples,
                sampler=sampler,
                custom_patterns=custom_patterns,
                seed_manager=seed,
            )
        )
        return self

    def is_consistent_across_samples(
        self,
        threshold: Optional[float] = None,
        n_samples: Optional[int] = None,
    ) -> AssertionBuilder:
        """Assert responses to the same prompt are semantically consistent.

        Runs the provider n_samples times with the same input, computes
        pairwise cosine similarity, and passes if average similarity
        exceeds threshold. Catches high-variance outputs from temperature
        instability or prompt fragility.

        Default threshold from config: 0.85. Default n_samples: 10.
        """
        seed = SeedManager(self._runner.config.seed)
        self._assertions.append(
            IsConsistentAcrossSamples(
                provider=self._runner.provider,
                threshold=threshold,
                n_samples=n_samples,
                seed_manager=seed,
            )
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
