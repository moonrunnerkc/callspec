"""Semantic assertions: meaning-based checks using embedding similarity.

All assertions in this module compute embeddings via sentence-transformers
and use cosine similarity for comparison. They are probabilistic by nature:
the same output might score 0.82 on one conceptual evaluation and 0.79 on
another due to embedding model variance across versions. Thresholds and
confidence intervals handle this.

Cost model: one embedding call per semantic assertion, approximately 5-20ms
on CPU for a typical response using all-MiniLM-L6-v2.
"""

from __future__ import annotations

from verdict.assertions.base import BaseAssertion
from verdict.core.config import VerdictConfig
from verdict.core.types import IndividualAssertionResult
from verdict.scoring.embeddings import score_similarity
from verdict.scoring.structural import compute_flesch_kincaid_grade


class SemanticIntentMatches(BaseAssertion):
    """Passes if the response is semantically aligned with a reference intent.

    Embeds both the response and the reference intent description, computes
    cosine similarity, and passes if similarity exceeds the threshold.

    Default threshold 0.75 is calibrated against SBERT STS-B benchmarks
    (Reimers & Gurevych, 2019): 0.75 cosine similarity in the all-MiniLM
    space corresponds to "clearly semantically related" per human annotators.
    """

    assertion_type = "semantic"
    assertion_name = "semantic_intent_matches"

    def __init__(
        self,
        reference_intent: str,
        threshold: float | None = None,
    ) -> None:
        self._reference_intent = reference_intent
        self._threshold = threshold

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        threshold = self._threshold if self._threshold is not None else config.semantic_similarity_threshold

        similarity = score_similarity(content, self._reference_intent, config.embedding_model)

        passed = similarity >= threshold

        if passed:
            message = (
                f"Semantic intent match: score {similarity:.4f} >= threshold {threshold:.4f} "
                f"using {config.embedding_model}. "
                f"Response aligns with intent: '{self._reference_intent[:80]}'"
            )
        else:
            message = (
                f"SemanticAssertion failed: score {similarity:.4f} below threshold {threshold:.4f} "
                f"using {config.embedding_model}, "
                f"input {len(content)} chars, "
                f"reference intent: '{self._reference_intent[:80]}'. "
                f"Check embedding model version or lower threshold if intent ambiguity is acceptable."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            score=similarity,
            threshold=threshold,
            details={
                "reference_intent": self._reference_intent,
                "embedding_model": config.embedding_model,
                "content_length": len(content),
            },
        )


class DoesNotDiscuss(BaseAssertion):
    """Passes if the response does NOT discuss a prohibited topic.

    Uses a lower threshold than positive matching (default 0.6) because
    you want to catch responses that approach the topic even loosely.
    """

    assertion_type = "semantic"
    assertion_name = "does_not_discuss"

    def __init__(
        self,
        topic: str,
        threshold: float | None = None,
    ) -> None:
        self._topic = topic
        self._threshold = threshold

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        threshold = self._threshold if self._threshold is not None else config.topic_avoidance_threshold

        similarity = score_similarity(content, self._topic, config.embedding_model)

        # Passes when similarity is BELOW threshold (topic is not discussed)
        passed = similarity < threshold

        if passed:
            message = (
                f"Topic avoidance check passed: similarity {similarity:.4f} < threshold {threshold:.4f}. "
                f"Response does not discuss: '{self._topic[:80]}'"
            )
        else:
            message = (
                f"SemanticAssertion failed: response appears to discuss prohibited topic. "
                f"Similarity {similarity:.4f} >= threshold {threshold:.4f} "
                f"using {config.embedding_model}. "
                f"Prohibited topic: '{self._topic[:80]}'. "
                f"Raise threshold if false positive, or modify prompt to avoid this topic."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            score=similarity,
            threshold=threshold,
            details={
                "topic": self._topic,
                "embedding_model": config.embedding_model,
                "content_length": len(content),
            },
        )


class IsFactuallyConsistentWith(BaseAssertion):
    """Passes if the response is semantically consistent with reference text.

    This is a consistency check, not a factual accuracy detector. It verifies
    that the response does not contradict or diverge significantly from the
    provided grounding material. Useful for RAG applications.

    Limitation: if the reference is wrong, a response that repeats the wrong
    information will pass. This measures consistency with a reference, not
    truth.
    """

    assertion_type = "semantic"
    assertion_name = "is_factually_consistent_with"

    def __init__(
        self,
        reference_text: str,
        threshold: float | None = None,
    ) -> None:
        self._reference_text = reference_text
        self._threshold = threshold

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        threshold = self._threshold if self._threshold is not None else config.factual_consistency_threshold

        similarity = score_similarity(content, self._reference_text, config.embedding_model)

        passed = similarity >= threshold

        # Truncate reference for message readability
        reference_preview = self._reference_text[:120]

        if passed:
            message = (
                f"Factual consistency check passed: score {similarity:.4f} >= threshold {threshold:.4f}. "
                f"Response is consistent with reference: '{reference_preview}...'"
            )
        else:
            message = (
                f"SemanticAssertion failed: factual consistency score {similarity:.4f} "
                f"below threshold {threshold:.4f} using {config.embedding_model}. "
                f"Response diverges from reference: '{reference_preview}...'. "
                f"Check that the model is using the provided context rather than generating from training data."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            score=similarity,
            threshold=threshold,
            details={
                "reference_text_length": len(self._reference_text),
                "embedding_model": config.embedding_model,
                "content_length": len(content),
            },
        )


class UsesLanguageAtGradeLevel(BaseAssertion):
    """Passes if response readability falls within the target grade range.

    Uses the Flesch-Kincaid grade level formula (validated since 1948).
    Purely arithmetic, deterministic, zero API cost. Not an embedding-based
    assertion despite living in the semantic module; it is a structural-semantic
    hybrid that evaluates language complexity.
    """

    assertion_type = "semantic"
    assertion_name = "uses_language_at_grade_level"

    def __init__(self, grade: int, tolerance: int = 2) -> None:
        self._target_grade = grade
        self._tolerance = tolerance

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        actual_grade = compute_flesch_kincaid_grade(content)
        lower_bound = self._target_grade - self._tolerance
        upper_bound = self._target_grade + self._tolerance

        passed = lower_bound <= actual_grade <= upper_bound

        if passed:
            message = (
                f"Grade level check passed: Flesch-Kincaid grade {actual_grade:.1f} "
                f"is within target range [{lower_bound}, {upper_bound}] "
                f"(target grade {self._target_grade} +/- {self._tolerance})."
            )
        else:
            direction = "below" if actual_grade < lower_bound else "above"
            message = (
                f"SemanticAssertion failed: Flesch-Kincaid grade {actual_grade:.1f} "
                f"is {direction} the target range [{lower_bound}, {upper_bound}]. "
                f"Target: grade {self._target_grade} +/- {self._tolerance}. "
                f"Adjust prompt instructions for {'simpler' if direction == 'above' else 'more complex'} language."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            score=actual_grade,
            threshold=float(self._target_grade),
            details={
                "actual_grade": actual_grade,
                "target_grade": self._target_grade,
                "tolerance": self._tolerance,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "content_length": len(content),
            },
        )
