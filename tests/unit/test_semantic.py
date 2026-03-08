"""Unit tests for all semantic assertion types.

Tests use known-good and known-bad pairs from the fixtures module.
Embedding-based tests load the all-MiniLM-L6-v2 model on first run
(22MB download on first execution; cached thereafter).

Grade level tests use Flesch-Kincaid (deterministic, no model needed).
"""

from __future__ import annotations

import pytest

from llm_assert.assertions.semantic import (
    DoesNotDiscuss,
    IsFactuallyConsistentWith,
    SemanticIntentMatches,
    UsesLanguageAtGradeLevel,
)
from llm_assert.core.config import LLMAssertConfig
from tests.fixtures.semantic_pairs import (
    FACTUAL_CONSISTENCY_BAD_PAIRS,
    FACTUAL_CONSISTENCY_GOOD_PAIRS,
    GRADE_LEVEL_TEXTS,
    INTENT_MATCH_BAD_PAIRS,
    INTENT_MATCH_GOOD_PAIRS,
    TOPIC_AVOIDANCE_BAD_PAIRS,
    TOPIC_AVOIDANCE_GOOD_PAIRS,
)

CONFIG = LLMAssertConfig()


# -- SemanticIntentMatches --


class TestSemanticIntentMatches:

    @pytest.mark.parametrize("pair", INTENT_MATCH_GOOD_PAIRS, ids=lambda p: p["reason"][:50])
    def test_good_pairs_pass(self, pair: dict) -> None:
        assertion = SemanticIntentMatches(pair["intent"])
        assertion_result = assertion.evaluate(pair["content"], CONFIG)
        assert assertion_result.passed is True, (
            f"Expected pass for good pair. Score: {assertion_result.score:.4f}, "
            f"Threshold: {assertion_result.threshold}, Reason: {pair['reason']}"
        )
        assert assertion_result.score is not None
        assert assertion_result.score >= 0.75

    @pytest.mark.parametrize("pair", INTENT_MATCH_BAD_PAIRS, ids=lambda p: p["reason"][:50])
    def test_bad_pairs_fail(self, pair: dict) -> None:
        assertion = SemanticIntentMatches(pair["intent"])
        assertion_result = assertion.evaluate(pair["content"], CONFIG)
        assert assertion_result.passed is False, (
            f"Expected fail for bad pair. Score: {assertion_result.score:.4f}, "
            f"Reason: {pair['reason']}"
        )

    def test_custom_threshold(self) -> None:
        assertion = SemanticIntentMatches(
            "a Python library that tests LLM behavior with behavioral assertions",
            threshold=0.5,
        )
        content = (
            "This Python library tests LLM behavior by running behavioral "
            "assertions against model output."
        )
        assertion_result = assertion.evaluate(content, CONFIG)
        assert assertion_result.passed is True
        assert assertion_result.threshold == 0.5

    def test_high_threshold_strict(self) -> None:
        """Very high threshold should reject loosely related content."""
        assertion = SemanticIntentMatches(
            "explain the rules of cricket",
            threshold=0.98,
        )
        content = "Cricket is a bat-and-ball game played between two teams."
        assertion_result = assertion.evaluate(content, CONFIG)
        # Even related content should fail at 0.98 unless nearly identical
        assert assertion_result.score is not None
        assert assertion_result.score < 0.98

    def test_identical_text_scores_high(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        assertion = SemanticIntentMatches(text, threshold=0.99)
        assertion_result = assertion.evaluate(text, CONFIG)
        assert assertion_result.passed is True
        assert assertion_result.score is not None
        assert assertion_result.score > 0.99

    def test_result_contains_metadata(self) -> None:
        assertion = SemanticIntentMatches("test intent")
        assertion_result = assertion.evaluate("test content", CONFIG)
        assert assertion_result.assertion_type == "semantic"
        assert assertion_result.assertion_name == "semantic_intent_matches"
        assert "embedding_model" in assertion_result.details
        assert "content_length" in assertion_result.details


# -- DoesNotDiscuss --


class TestDoesNotDiscuss:

    @pytest.mark.parametrize("pair", TOPIC_AVOIDANCE_GOOD_PAIRS, ids=lambda p: p["reason"][:50])
    def test_good_avoidance_passes(self, pair: dict) -> None:
        assertion = DoesNotDiscuss(pair["topic"])
        assertion_result = assertion.evaluate(pair["content"], CONFIG)
        assert assertion_result.passed is True, (
            f"Expected pass (topic avoided). Score: {assertion_result.score:.4f}, "
            f"Reason: {pair['reason']}"
        )

    @pytest.mark.parametrize("pair", TOPIC_AVOIDANCE_BAD_PAIRS, ids=lambda p: p["reason"][:50])
    def test_bad_avoidance_fails(self, pair: dict) -> None:
        assertion = DoesNotDiscuss(pair["topic"])
        assertion_result = assertion.evaluate(pair["content"], CONFIG)
        assert assertion_result.passed is False, (
            f"Expected fail (topic present). Score: {assertion_result.score:.4f}, "
            f"Reason: {pair['reason']}"
        )

    def test_completely_unrelated_topic_passes(self) -> None:
        assertion = DoesNotDiscuss("underwater basket weaving techniques")
        content = "The server responds with a 200 OK status code and a JSON body."
        assertion_result = assertion.evaluate(content, CONFIG)
        assert assertion_result.passed is True

    def test_custom_threshold(self) -> None:
        assertion = DoesNotDiscuss("technology", threshold=0.9)
        content = "This software runs on Linux servers."
        # With a very high threshold (0.9), even loosely tech-related content passes
        assertion_result = assertion.evaluate(content, CONFIG)
        assert assertion_result.threshold == 0.9


# -- IsFactuallyConsistentWith --


class TestIsFactuallyConsistentWith:

    @pytest.mark.parametrize(
        "pair", FACTUAL_CONSISTENCY_GOOD_PAIRS, ids=lambda p: p["reason"][:50]
    )
    def test_consistent_pairs_pass(self, pair: dict) -> None:
        assertion = IsFactuallyConsistentWith(pair["reference"])
        assertion_result = assertion.evaluate(pair["content"], CONFIG)
        assert assertion_result.passed is True, (
            f"Expected pass (consistent). Score: {assertion_result.score:.4f}, "
            f"Reason: {pair['reason']}"
        )

    @pytest.mark.parametrize(
        "pair", FACTUAL_CONSISTENCY_BAD_PAIRS, ids=lambda p: p["reason"][:50]
    )
    def test_inconsistent_pairs_fail(self, pair: dict) -> None:
        assertion = IsFactuallyConsistentWith(pair["reference"])
        assertion_result = assertion.evaluate(pair["content"], CONFIG)
        assert assertion_result.passed is False, (
            f"Expected fail (inconsistent). Score: {assertion_result.score:.4f}, "
            f"Reason: {pair['reason']}"
        )

    def test_identical_reference_passes(self) -> None:
        reference = "The Earth orbits the Sun at an average distance of 93 million miles."
        assertion = IsFactuallyConsistentWith(reference)
        assertion_result = assertion.evaluate(reference, CONFIG)
        assert assertion_result.passed is True
        assert assertion_result.score is not None
        assert assertion_result.score > 0.99

    def test_result_includes_reference_length(self) -> None:
        reference = "Short reference."
        assertion = IsFactuallyConsistentWith(reference)
        assertion_result = assertion.evaluate("Some content.", CONFIG)
        assert "reference_text_length" in assertion_result.details


# -- UsesLanguageAtGradeLevel --


class TestUsesLanguageAtGradeLevel:

    def test_simple_text_low_grade(self) -> None:
        pair = GRADE_LEVEL_TEXTS[0]
        # Simple text should be within grade 0-4 (target 2, tolerance 2)
        assertion = UsesLanguageAtGradeLevel(grade=2, tolerance=2)
        assertion_result = assertion.evaluate(pair["text"], CONFIG)
        assert assertion_result.passed is True, (
            f"Expected simple text in grade range [0, 4]. "
            f"Actual grade: {assertion_result.score:.1f}"
        )

    def test_complex_text_high_grade(self) -> None:
        pair = GRADE_LEVEL_TEXTS[1]
        # Academic text should be high grade level
        assertion = UsesLanguageAtGradeLevel(grade=20, tolerance=10)
        assertion_result = assertion.evaluate(pair["text"], CONFIG)
        assert assertion_result.passed is True, (
            f"Expected complex text in grade range [10, 30]. "
            f"Actual grade: {assertion_result.score:.1f}"
        )

    def test_grade_too_high_fails(self) -> None:
        simple_text = "The cat sat on the mat. It was a good cat."
        assertion = UsesLanguageAtGradeLevel(grade=16, tolerance=2)
        assertion_result = assertion.evaluate(simple_text, CONFIG)
        assert assertion_result.passed is False

    def test_grade_too_low_fails(self) -> None:
        complex_text = GRADE_LEVEL_TEXTS[1]["text"]
        assertion = UsesLanguageAtGradeLevel(grade=3, tolerance=2)
        assertion_result = assertion.evaluate(complex_text, CONFIG)
        assert assertion_result.passed is False

    def test_empty_text(self) -> None:
        assertion = UsesLanguageAtGradeLevel(grade=5, tolerance=5)
        assertion_result = assertion.evaluate("", CONFIG)
        # Empty text produces grade 0.0, which is within [0, 10]
        assert assertion_result.score == 0.0

    def test_tolerance_range(self) -> None:
        """Verify tolerance expands the acceptable range symmetrically."""
        assertion = UsesLanguageAtGradeLevel(grade=8, tolerance=3)
        assertion_result = assertion.evaluate("Some text here.", CONFIG)
        assert assertion_result.details["lower_bound"] == 5
        assert assertion_result.details["upper_bound"] == 11

    def test_assertion_metadata(self) -> None:
        assertion = UsesLanguageAtGradeLevel(grade=10, tolerance=2)
        assertion_result = assertion.evaluate("Test text.", CONFIG)
        assert assertion_result.assertion_type == "semantic"
        assert assertion_result.assertion_name == "uses_language_at_grade_level"
        assert assertion_result.threshold == 10.0
