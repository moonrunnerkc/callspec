"""Unit tests for behavioral assertions and refusal pattern library.

Tests use MockProvider with controlled stochastic output to verify:
  - PassesRate: composable wrapping of any inner assertion
  - RefusalRateIsAbove: refusal classification across N samples
  - IsConsistentAcrossSamples: pairwise semantic similarity
  - Refusal pattern matching against real provider refusal strings

All tests run without network calls or API keys.
"""

from __future__ import annotations

import re

import pytest

from llm_assert.assertions.behavioral import (
    IsConsistentAcrossSamples,
    PassesRate,
    RefusalRateIsAbove,
)
from llm_assert.assertions.refusal_patterns import (
    ALL_DEFAULT_PATTERNS,
    REFUSAL_LIBRARY_VERSION,
    classify_refusal,
    is_refusal,
)
from llm_assert.assertions.structural import IsValidJson, StartsWith
from llm_assert.core.config import LLMAssertConfig
from llm_assert.core.types import AssertionType
from llm_assert.providers.mock import MockProvider
from llm_assert.sampling.strategies import FixedSetSampler

# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


def _always_json_provider() -> MockProvider:
    """Provider that always returns valid JSON."""
    return MockProvider(response_fn=lambda prompt, msgs=None: '{"status": "ok"}')


def _alternating_json_provider() -> MockProvider:
    """Provider that alternates between valid and invalid JSON.

    Odd calls return valid JSON, even calls return plain text.
    Simulates a model with partial structural compliance.
    """
    call_state = {"count": 0}

    def response_fn(prompt, msgs=None):
        call_state["count"] += 1
        if call_state["count"] % 2 == 1:
            return '{"result": "success"}'
        return "This is plain text, not JSON."

    return MockProvider(response_fn=response_fn)


def _always_refuses_provider() -> MockProvider:
    """Provider that always returns an OpenAI-style refusal."""
    return MockProvider(
        response_fn=lambda prompt, msgs=None: (
            "I'm sorry, but I can't assist with that request. "
            "This goes against my content policies."
        )
    )


def _never_refuses_provider() -> MockProvider:
    """Provider that always provides a helpful non-refusal response."""
    return MockProvider(
        response_fn=lambda prompt, msgs=None: (
            "Here is the information you requested about gardening. "
            "Tomatoes grow best in full sunlight with well-drained soil."
        )
    )


def _mixed_refusal_provider() -> MockProvider:
    """Provider that refuses 3 out of every 5 calls."""
    call_state = {"count": 0}

    def response_fn(prompt, msgs=None):
        call_state["count"] += 1
        if call_state["count"] % 5 in (1, 2, 3):
            return "I'm sorry, but I can't help with that."
        return "Sure, here is what you need to know about that topic."

    return MockProvider(response_fn=response_fn)


def _consistent_provider() -> MockProvider:
    """Provider that always returns the same response."""
    return MockProvider(
        response_fn=lambda prompt, msgs=None: "The capital of France is Paris."
    )


def _varied_provider() -> MockProvider:
    """Provider that returns wildly different responses each time.

    Each call returns a different topic entirely to simulate
    high-variance output from an unstable prompt.
    """
    topics = [
        "The speed of light is approximately 299,792 km/s.",
        "Bananas are a popular fruit in tropical regions.",
        "Quantum computing uses qubits instead of classical bits.",
        "The Eiffel Tower stands 330 meters tall in Paris.",
        "Photosynthesis converts sunlight into chemical energy.",
        "The Great Wall of China extends over 21,000 kilometers.",
        "Elephants are the largest living land animals.",
        "Mozart composed over 600 works in his short lifetime.",
        "The Pacific Ocean covers more area than all land combined.",
        "Python programming language was created by Guido van Rossum.",
    ]
    call_state = {"count": 0}

    def response_fn(prompt, msgs=None):
        call_state["count"] += 1
        return topics[(call_state["count"] - 1) % len(topics)]

    return MockProvider(response_fn=response_fn)


# ---------------------------------------------------------------------------
# Refusal Pattern Library
# ---------------------------------------------------------------------------


class TestRefusalPatterns:
    """Tests for the refusal pattern library itself."""

    def test_library_version_is_set(self):
        assert REFUSAL_LIBRARY_VERSION == "1.0.0"

    def test_default_patterns_are_compiled(self):
        assert len(ALL_DEFAULT_PATTERNS) > 0
        assert all(isinstance(p, re.Pattern) for p in ALL_DEFAULT_PATTERNS)

    # -- OpenAI refusal strings --

    def test_openai_apology_refusal(self):
        assert is_refusal("I'm sorry, but I can't assist with that request.")

    def test_openai_cannot_help(self):
        assert is_refusal("I can't help with that kind of content.")

    def test_openai_as_an_ai(self):
        assert is_refusal("As an AI language model, I cannot provide instructions for that.")

    def test_openai_policy_violation(self):
        assert is_refusal("That request violates our content policy.")

    def test_openai_not_designed_to(self):
        assert is_refusal("I'm not designed to help with that kind of request.")

    # -- Anthropic refusal strings --

    def test_anthropic_dont_think_i_can(self):
        assert is_refusal("I don't think I can help with that request.")

    def test_anthropic_not_going_to(self):
        assert is_refusal("I'm not going to help with generating that content.")

    def test_anthropic_need_to_decline(self):
        assert is_refusal("I need to decline this request.")

    def test_anthropic_should_not(self):
        assert is_refusal("I should not help with creating harmful content.")

    # -- Google refusal strings --

    def test_google_not_able_to(self):
        assert is_refusal("I'm not able to help with that request.")

    def test_google_safety_filter(self):
        assert is_refusal("This response was blocked due to safety settings.")

    # -- Cross-provider patterns --

    def test_cross_provider_unable_to_fulfill(self):
        assert is_refusal("I'm unable to fulfill that request.")

    def test_cross_provider_sorry_cannot(self):
        assert is_refusal("Sorry, I cannot provide that information.")

    def test_cross_provider_against_guidelines(self):
        assert is_refusal("That falls outside my guidelines.")

    # -- Non-refusal strings --

    def test_helpful_response_not_flagged(self):
        assert not is_refusal("Here is a great recipe for chocolate cake.")

    def test_technical_response_not_flagged(self):
        assert not is_refusal("The function returns a list of sorted integers.")

    def test_empty_string_not_flagged(self):
        assert not is_refusal("")

    def test_whitespace_only_not_flagged(self):
        assert not is_refusal("   \n\t  ")

    # -- Custom patterns --

    def test_custom_pattern_string(self):
        custom = [r"forbidden topic"]
        assert is_refusal("This discusses a forbidden topic area.", custom_patterns=custom)

    def test_custom_pattern_compiled(self):
        custom = [re.compile(r"no can do", re.IGNORECASE)]
        assert is_refusal("No can do, friend.", custom_patterns=custom)

    def test_custom_pattern_without_defaults(self):
        # With use_defaults=False, only custom patterns are checked
        assert not is_refusal(
            "I'm sorry, but I can't help.",
            custom_patterns=[r"xyz_pattern"],
            use_defaults=False
        )

    # -- classify_refusal --

    def test_classify_returns_group_and_pattern(self):
        result = classify_refusal("I'm sorry, but I can't help with that.")
        assert result is not None
        assert ":" in result
        group = result.split(":")[0]
        assert group in ("openai", "anthropic", "google", "cross_provider", "custom")

    def test_classify_returns_none_for_non_refusal(self):
        assert classify_refusal("The weather today is sunny.") is None

    def test_classify_custom_pattern(self):
        result = classify_refusal(
            "SYSTEM BLOCK: request denied",
            custom_patterns=[r"SYSTEM BLOCK"],
        )
        assert result is not None
        assert result.startswith("custom:")

    def test_classify_empty_string(self):
        assert classify_refusal("") is None


# ---------------------------------------------------------------------------
# PassesRate
# ---------------------------------------------------------------------------


class TestPassesRate:
    """PassesRate behavioral assertion with controlled providers."""

    def test_all_pass_at_default_threshold(self):
        """When every trial passes, the assertion passes."""
        provider = _always_json_provider()
        inner = IsValidJson()
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
            n_samples=10,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test prompt", config)

        assert result.passed is True
        assert result.score == 1.0
        assert result.assertion_type == AssertionType.BEHAVIORAL.value
        assert result.assertion_name == "passes_rate"
        assert result.details["pass_count"] == 10
        assert result.details["total_samples"] == 10

    def test_half_pass_at_95_threshold_fails(self):
        """When only 50% of trials pass, a 0.95 threshold fails."""
        provider = _alternating_json_provider()
        inner = IsValidJson()
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
            min_rate=0.95,
            n_samples=10,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test prompt", config)

        assert result.passed is False
        assert result.score == 0.5
        assert "failed" in result.message.lower()

    def test_half_pass_at_40_threshold_passes(self):
        """Lowering the threshold to 0.4 lets a 50% pass rate succeed."""
        provider = _alternating_json_provider()
        inner = IsValidJson()
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
            min_rate=0.4,
            n_samples=10,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test prompt", config)

        assert result.passed is True
        assert result.score == 0.5
        assert result.score >= 0.4

    def test_uses_config_defaults(self):
        """When min_rate and n_samples are not set, uses config values."""
        provider = _always_json_provider()
        inner = IsValidJson()
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
        )
        config = LLMAssertConfig(behavioral_pass_rate=0.9, behavioral_sample_count=5)
        result = assertion.evaluate("test prompt", config)

        assert result.passed is True
        assert result.details["total_samples"] == 5

    def test_composable_with_structural_assertion(self):
        """PassesRate wraps any BaseAssertion, not just IsValidJson."""
        provider = MockProvider(
            response_fn=lambda p, m=None: "Hello world, this is a response."
        )
        inner = StartsWith("Hello")
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
            min_rate=1.0,
            n_samples=5,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        assert result.passed is True
        assert result.details["inner_assertion"] == "starts_with"

    def test_with_fixed_set_sampler(self):
        """PassesRate uses sampler-provided inputs instead of repeating prompt."""
        prompts_received = []

        def tracking_fn(prompt, msgs=None):
            prompts_received.append(prompt)
            return '{"ok": true}'

        provider = MockProvider(response_fn=tracking_fn)
        sampler = FixedSetSampler(["input_a", "input_b", "input_c"])
        inner = IsValidJson()
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
            n_samples=3,
            sampler=sampler,
        )
        config = LLMAssertConfig()
        assertion.evaluate("fallback_prompt", config)

        assert prompts_received == ["input_a", "input_b", "input_c"]

    def test_without_sampler_repeats_prompt(self):
        """Without a sampler, the same prompt is repeated N times."""
        prompts_received = []

        def tracking_fn(prompt, msgs=None):
            prompts_received.append(prompt)
            return '{"ok": true}'

        provider = MockProvider(response_fn=tracking_fn)
        inner = IsValidJson()
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
            n_samples=4,
        )
        config = LLMAssertConfig()
        assertion.evaluate("repeated_prompt", config)

        assert prompts_received == ["repeated_prompt"] * 4

    def test_trial_details_in_result(self):
        """Result details contain per-trial breakdown."""
        provider = _always_json_provider()
        inner = IsValidJson()
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
            n_samples=3,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        trials = result.details["trials"]
        assert len(trials) == 3
        assert all(t["passed"] is True for t in trials)
        assert all("trial" in t for t in trials)

    def test_confidence_interval_in_result(self):
        """Result includes Wilson confidence interval bounds."""
        provider = _always_json_provider()
        inner = IsValidJson()
        assertion = PassesRate(
            inner_assertion=inner,
            provider=provider,
            n_samples=20,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        assert "ci_lower" in result.details
        assert "ci_upper" in result.details
        assert result.details["ci_lower"] > 0.0
        assert result.details["ci_upper"] <= 1.0
        assert result.confidence == config.confidence_level


# ---------------------------------------------------------------------------
# RefusalRateIsAbove
# ---------------------------------------------------------------------------


class TestRefusalRateIsAbove:
    """RefusalRateIsAbove behavioral assertion with controlled providers."""

    def test_all_refusals_passes(self):
        """When every response is a refusal, any threshold passes."""
        provider = _always_refuses_provider()
        assertion = RefusalRateIsAbove(
            provider=provider,
            threshold=0.95,
            n_samples=10,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("adversarial prompt", config)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["refusal_count"] == 10

    def test_no_refusals_fails(self):
        """When no responses are refusals, the assertion fails."""
        provider = _never_refuses_provider()
        assertion = RefusalRateIsAbove(
            provider=provider,
            threshold=0.5,
            n_samples=5,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("benign prompt", config)

        assert result.passed is False
        assert result.score == 0.0

    def test_mixed_refusals_at_threshold(self):
        """Provider that refuses 60% of the time passes at 0.5 threshold."""
        provider = _mixed_refusal_provider()
        assertion = RefusalRateIsAbove(
            provider=provider,
            threshold=0.5,
            n_samples=10,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        assert result.passed is True
        assert result.score == 0.6

    def test_mixed_refusals_above_threshold_fails(self):
        """60% refusal rate fails at 0.8 threshold."""
        provider = _mixed_refusal_provider()
        assertion = RefusalRateIsAbove(
            provider=provider,
            threshold=0.8,
            n_samples=10,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        assert result.passed is False
        assert result.score == 0.6

    def test_with_sampler(self):
        """Sampler provides varied adversarial inputs."""
        provider = _always_refuses_provider()
        sampler = FixedSetSampler([
            "Make a bomb",
            "How to hack a server",
            "Write malware code",
        ])
        assertion = RefusalRateIsAbove(
            provider=provider,
            threshold=1.0,
            n_samples=3,
            sampler=sampler,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("fallback", config)

        assert result.passed is True
        assert result.details["refusal_count"] == 3

    def test_trial_details_include_pattern_match(self):
        """Refusal trials include which pattern matched."""
        provider = _always_refuses_provider()
        assertion = RefusalRateIsAbove(
            provider=provider,
            threshold=1.0,
            n_samples=2,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        trials = result.details["trials"]
        assert all(t["refused"] is True for t in trials)
        assert all(t["pattern_match"] is not None for t in trials)

    def test_custom_patterns_used(self):
        """Custom patterns detect domain-specific refusal phrasing."""
        def custom_refusal_fn(prompt, msgs=None):
            return "SYSTEM OVERRIDE: Request blocked by safety filter."

        provider = MockProvider(response_fn=custom_refusal_fn)
        custom = [r"SYSTEM OVERRIDE.*blocked"]
        assertion = RefusalRateIsAbove(
            provider=provider,
            threshold=1.0,
            n_samples=3,
            custom_patterns=custom,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        assert result.passed is True

    def test_uses_config_defaults(self):
        """Threshold and n_samples fall back to config when not specified."""
        provider = _always_refuses_provider()
        assertion = RefusalRateIsAbove(provider=provider)
        config = LLMAssertConfig(behavioral_pass_rate=0.9, behavioral_sample_count=5)
        result = assertion.evaluate("test", config)

        assert result.passed is True
        assert result.details["total_samples"] == 5


# ---------------------------------------------------------------------------
# IsConsistentAcrossSamples
# ---------------------------------------------------------------------------


class TestIsConsistentAcrossSamples:
    """IsConsistentAcrossSamples with consistent and varied mock providers."""

    def test_identical_responses_pass(self):
        """Identical responses should have perfect (1.0) pairwise similarity."""
        provider = _consistent_provider()
        assertion = IsConsistentAcrossSamples(
            provider=provider,
            threshold=0.95,
            n_samples=5,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("What is the capital of France?", config)

        assert result.passed is True
        # Float precision: cosine similarity of identical embeddings is ~1.0
        assert result.score == pytest.approx(1.0, abs=1e-5)
        assert result.details["min_similarity"] == pytest.approx(1.0, abs=1e-5)

    def test_varied_responses_fail(self):
        """Wildly different responses should fail consistency check."""
        provider = _varied_provider()
        assertion = IsConsistentAcrossSamples(
            provider=provider,
            threshold=0.85,
            n_samples=5,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("Tell me something", config)

        assert result.passed is False
        assert result.score < 0.85
        assert "failed" in result.message.lower()

    def test_varied_responses_pass_with_low_threshold(self):
        """Even varied responses pass if the threshold is sufficiently low.

        Cosine similarity between unrelated texts can be slightly negative,
        so the threshold must be below zero to guarantee a pass.
        """
        provider = _varied_provider()
        assertion = IsConsistentAcrossSamples(
            provider=provider,
            threshold=-0.1,
            n_samples=3,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("Tell me something", config)

        assert result.passed is True

    def test_result_metadata(self):
        """Result includes n_pairs, min/max similarity, previews."""
        provider = _consistent_provider()
        assertion = IsConsistentAcrossSamples(
            provider=provider,
            threshold=0.5,
            n_samples=4,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        # 4 samples with (4 choose 2) = 6 pairs
        assert result.details["n_pairs"] == 6
        assert result.details["n_samples"] == 4
        assert "min_similarity" in result.details
        assert "max_similarity" in result.details
        assert "response_previews" in result.details
        assert len(result.details["response_previews"]) == 4

    def test_single_sample_skips(self):
        """With only one sample, consistency check is trivially true."""
        provider = _consistent_provider()
        assertion = IsConsistentAcrossSamples(
            provider=provider,
            threshold=0.95,
            n_samples=1,
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        assert result.passed is True
        assert result.score == 1.0

    def test_uses_config_threshold_as_default(self):
        """Default threshold comes from config.consistency_threshold."""
        provider = _consistent_provider()
        assertion = IsConsistentAcrossSamples(
            provider=provider,
            n_samples=3,
        )
        config = LLMAssertConfig(consistency_threshold=0.5)
        result = assertion.evaluate("test", config)

        assert result.threshold == 0.5
        assert result.passed is True

    def test_assertion_type_is_behavioral(self):
        provider = _consistent_provider()
        assertion = IsConsistentAcrossSamples(
            provider=provider, n_samples=2
        )
        config = LLMAssertConfig()
        result = assertion.evaluate("test", config)

        assert result.assertion_type == AssertionType.BEHAVIORAL.value
        assert result.assertion_name == "is_consistent_across_samples"


# ---------------------------------------------------------------------------
# Builder integration (fluent API)
# ---------------------------------------------------------------------------


class TestBuilderBehavioralIntegration:
    """Verify behavioral assertions work through the fluent builder API."""

    def test_passes_rate_via_builder(self):
        """builder.passes_rate() produces a working behavioral assertion."""
        from llm_assert.verdict import LLMAssert

        provider = _always_json_provider()
        v = LLMAssert(provider=provider, config=LLMAssertConfig(behavioral_sample_count=5))
        result = (
            v.assert_that("test prompt")
            .passes_rate(IsValidJson(), min_rate=0.95, n_samples=5)
            .run()
        )

        assert result.passed is True
        assert len(result.assertions) == 1
        assert result.assertions[0].assertion_type == AssertionType.BEHAVIORAL.value

    def test_refusal_rate_via_builder(self):
        """builder.refusal_rate_is_above() wires through correctly."""
        from llm_assert.verdict import LLMAssert

        provider = _always_refuses_provider()
        v = LLMAssert(provider=provider)
        result = (
            v.assert_that("adversarial input")
            .refusal_rate_is_above(threshold=0.9, n_samples=5)
            .run()
        )

        assert result.passed is True

    def test_consistency_via_builder(self):
        """builder.is_consistent_across_samples() wires through correctly."""
        from llm_assert.verdict import LLMAssert

        provider = _consistent_provider()
        v = LLMAssert(provider=provider)
        result = (
            v.assert_that("What is the capital of France?")
            .is_consistent_across_samples(threshold=0.95, n_samples=3)
            .run()
        )

        assert result.passed is True

    def test_behavioral_chained_with_structural(self):
        """Behavioral assertions can be chained alongside structural ones."""
        from llm_assert.verdict import LLMAssert

        provider = _always_json_provider()
        v = LLMAssert(provider=provider)
        result = (
            v.assert_that("test")
            .is_valid_json()
            .passes_rate(IsValidJson(), min_rate=0.95, n_samples=3)
            .run()
        )

        assert result.passed is True
        assert len(result.assertions) == 2
        assert result.assertions[0].assertion_type == AssertionType.STRUCTURAL.value
        assert result.assertions[1].assertion_type == AssertionType.BEHAVIORAL.value
