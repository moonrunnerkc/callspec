"""Behavioral assertions: verify patterns across multiple LLM outputs.

Behavioral assertions are fundamentally different from structural, semantic,
and regression assertions: they run the provider N times and assess the
distribution of outputs, not a single response. This makes them more
expensive and statistically more meaningful.

Three built-in behavioral assertions:

  PassesRate  - wraps any inner assertion, passes if min_rate of N samples pass
  RefusalRateIsAbove - confirms the model reliably refuses a class of input
  IsConsistentAcrossSamples - measures pairwise semantic similarity variance

Behavioral assertions are composable: PassesRate wraps any BaseAssertion,
so you do not need a new assertion type for every behavioral check.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence

from verdict.assertions.base import BaseAssertion
from verdict.assertions.refusal_patterns import classify_refusal, is_refusal
from verdict.core.config import VerdictConfig
from verdict.core.types import AssertionType, IndividualAssertionResult, ProviderResponse
from verdict.providers.base import BaseProvider
from verdict.sampling.sampler import BaseSampler, InputItem
from verdict.sampling.seed import SeedManager
from verdict.scoring.confidence import wilson_confidence_interval

logger = logging.getLogger(__name__)


class PassesRate(BaseAssertion):
    """Behavioral assertion: inner assertion must pass at min_rate across N samples.

    Runs the provider n_samples times (using the sampler for varied inputs
    or repeating the prompt if no sampler is provided), applies the inner
    assertion to each output, and passes if the proportion of passing
    outputs meets or exceeds min_rate.

    The composability is the key design decision: any existing assertion
    can be wrapped with PassesRate to become a behavioral assertion.
    """

    assertion_type = AssertionType.BEHAVIORAL.value
    assertion_name = "passes_rate"

    def __init__(
        self,
        inner_assertion: BaseAssertion,
        provider: BaseProvider,
        min_rate: Optional[float] = None,
        n_samples: Optional[int] = None,
        sampler: Optional[BaseSampler] = None,
        seed_manager: Optional[SeedManager] = None,
    ) -> None:
        self._inner_assertion = inner_assertion
        self._provider = provider
        self._min_rate = min_rate
        self._n_samples = n_samples
        self._sampler = sampler
        self._seed_manager = seed_manager or SeedManager()

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        """Run multi-sample evaluation.

        The `content` parameter is the prompt text (used when no sampler
        is provided). When a sampler is configured, it generates the inputs.
        """
        min_rate = self._min_rate if self._min_rate is not None else config.behavioral_pass_rate
        n_samples = self._n_samples if self._n_samples is not None else config.behavioral_sample_count

        inputs = self._get_inputs(content, n_samples)
        pass_count = 0
        trial_details = []

        for trial_index, input_item in enumerate(inputs):
            response = self._provider.call(
                prompt=input_item.prompt,
                messages=input_item.messages,
                temperature=config.temperature,
                seed=config.seed,
            )

            trial_result = self._inner_assertion.evaluate(response.content, config)
            if trial_result.passed:
                pass_count += 1

            trial_details.append({
                "trial": trial_index + 1,
                "passed": trial_result.passed,
                "input_preview": input_item.prompt[:80],
                "output_preview": response.content[:80],
                "score": trial_result.score,
            })

        observed_rate = pass_count / n_samples if n_samples > 0 else 0.0
        lower_bound, upper_bound = wilson_confidence_interval(
            pass_count, n_samples, config.confidence_level
        )

        passed = observed_rate >= min_rate

        if passed:
            message = (
                f"PassesRate passed: {pass_count}/{n_samples} trials passed "
                f"(rate {observed_rate:.3f} >= threshold {min_rate:.3f}), "
                f"95% CI [{lower_bound:.3f}, {upper_bound:.3f}], "
                f"inner assertion: {self._inner_assertion.assertion_name}"
            )
        else:
            message = (
                f"PassesRate failed: {pass_count}/{n_samples} trials passed "
                f"(rate {observed_rate:.3f} < threshold {min_rate:.3f}), "
                f"95% CI [{lower_bound:.3f}, {upper_bound:.3f}], "
                f"inner assertion: {self._inner_assertion.assertion_name}. "
                f"Increase n_samples for narrower confidence interval or "
                f"lower min_rate if behavioral variance is acceptable."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            score=observed_rate,
            threshold=min_rate,
            confidence=config.confidence_level,
            details={
                "pass_count": pass_count,
                "total_samples": n_samples,
                "observed_rate": observed_rate,
                "ci_lower": lower_bound,
                "ci_upper": upper_bound,
                "inner_assertion": self._inner_assertion.assertion_name,
                "trials": trial_details,
            },
        )

    def _get_inputs(self, fallback_prompt: str, n: int) -> List[InputItem]:
        """Get inputs from the sampler, or repeat the prompt if no sampler."""
        if self._sampler is not None:
            return self._sampler.sample(n, self._seed_manager)
        return [InputItem(prompt=fallback_prompt)] * n


class RefusalRateIsAbove(BaseAssertion):
    """Behavioral assertion: model must refuse at least threshold of inputs.

    Runs the provider n_samples times against inputs from the sampler,
    classifies each response as a refusal or not, and passes if the
    refusal rate meets the threshold.

    Refusal classification uses Verdict's built-in refusal pattern library
    (covering OpenAI, Anthropic, and Google phrasings) plus any custom
    patterns the developer provides.
    """

    assertion_type = AssertionType.BEHAVIORAL.value
    assertion_name = "refusal_rate_is_above"

    def __init__(
        self,
        provider: BaseProvider,
        threshold: Optional[float] = None,
        n_samples: Optional[int] = None,
        sampler: Optional[BaseSampler] = None,
        custom_patterns: Optional[Sequence[str | re.Pattern]] = None,
        seed_manager: Optional[SeedManager] = None,
    ) -> None:
        self._provider = provider
        self._threshold = threshold
        self._n_samples = n_samples
        self._sampler = sampler
        self._custom_patterns = custom_patterns
        self._seed_manager = seed_manager or SeedManager()

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        threshold = self._threshold if self._threshold is not None else config.behavioral_pass_rate
        n_samples = self._n_samples if self._n_samples is not None else config.behavioral_sample_count

        inputs = self._get_inputs(content, n_samples)
        refusal_count = 0
        trial_details = []

        for trial_index, input_item in enumerate(inputs):
            response = self._provider.call(
                prompt=input_item.prompt,
                messages=input_item.messages,
                temperature=config.temperature,
                seed=config.seed,
            )

            refused = is_refusal(response.content, self._custom_patterns)
            pattern_match = classify_refusal(response.content, self._custom_patterns) if refused else None

            if refused:
                refusal_count += 1

            trial_details.append({
                "trial": trial_index + 1,
                "refused": refused,
                "pattern_match": pattern_match,
                "input_preview": input_item.prompt[:80],
                "output_preview": response.content[:80],
            })

        observed_rate = refusal_count / n_samples if n_samples > 0 else 0.0
        lower_bound, upper_bound = wilson_confidence_interval(
            refusal_count, n_samples, config.confidence_level
        )

        passed = observed_rate >= threshold

        if passed:
            message = (
                f"RefusalRateIsAbove passed: {refusal_count}/{n_samples} responses "
                f"were refusals (rate {observed_rate:.3f} >= threshold {threshold:.3f}), "
                f"95% CI [{lower_bound:.3f}, {upper_bound:.3f}]"
            )
        else:
            message = (
                f"RefusalRateIsAbove failed: {refusal_count}/{n_samples} responses "
                f"were refusals (rate {observed_rate:.3f} < threshold {threshold:.3f}), "
                f"95% CI [{lower_bound:.3f}, {upper_bound:.3f}]. "
                f"The model is not reliably refusing this input class."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            score=observed_rate,
            threshold=threshold,
            confidence=config.confidence_level,
            details={
                "refusal_count": refusal_count,
                "total_samples": n_samples,
                "observed_rate": observed_rate,
                "ci_lower": lower_bound,
                "ci_upper": upper_bound,
                "trials": trial_details,
            },
        )

    def _get_inputs(self, fallback_prompt: str, n: int) -> List[InputItem]:
        if self._sampler is not None:
            return self._sampler.sample(n, self._seed_manager)
        return [InputItem(prompt=fallback_prompt)] * n


class IsConsistentAcrossSamples(BaseAssertion):
    """Behavioral assertion: pairwise semantic similarity across repeated calls.

    Runs the provider n_samples times with the same prompt and measures
    pairwise cosine similarity between all response pairs. Passes if the
    average similarity exceeds the threshold.

    Catches high-variance outputs caused by temperature instability or
    prompt fragility. A model that produces wildly different responses
    to the same input is not reliable enough for production.
    """

    assertion_type = AssertionType.BEHAVIORAL.value
    assertion_name = "is_consistent_across_samples"

    def __init__(
        self,
        provider: BaseProvider,
        threshold: Optional[float] = None,
        n_samples: Optional[int] = None,
        seed_manager: Optional[SeedManager] = None,
    ) -> None:
        self._provider = provider
        self._threshold = threshold
        self._n_samples = n_samples
        self._seed_manager = seed_manager or SeedManager()

    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        threshold = self._threshold if self._threshold is not None else config.consistency_threshold
        n_samples = self._n_samples if self._n_samples is not None else 10

        # Collect N responses to the same prompt
        responses: List[str] = []
        for _ in range(n_samples):
            response = self._provider.call(
                prompt=content,
                temperature=config.temperature,
                seed=config.seed,
            )
            responses.append(response.content)

        if len(responses) < 2:
            return IndividualAssertionResult(
                assertion_type=self.assertion_type,
                assertion_name=self.assertion_name,
                passed=True,
                message="Consistency check skipped: fewer than 2 samples collected.",
                score=1.0,
                threshold=threshold,
            )

        # Compute pairwise semantic similarity
        from verdict.scoring.embeddings import compute_embeddings, cosine_similarity

        embeddings = compute_embeddings(responses, config.embedding_model)

        pair_similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                pair_similarities.append(similarity)

        average_similarity = sum(pair_similarities) / len(pair_similarities)
        min_similarity = min(pair_similarities)
        max_similarity = max(pair_similarities)

        passed = average_similarity >= threshold

        if passed:
            message = (
                f"IsConsistentAcrossSamples passed: average pairwise similarity "
                f"{average_similarity:.3f} >= threshold {threshold:.3f} "
                f"across {n_samples} samples "
                f"(min={min_similarity:.3f}, max={max_similarity:.3f})"
            )
        else:
            message = (
                f"IsConsistentAcrossSamples failed: average pairwise similarity "
                f"{average_similarity:.3f} < threshold {threshold:.3f} "
                f"across {n_samples} samples "
                f"(min={min_similarity:.3f}, max={max_similarity:.3f}). "
                f"High variance suggests prompt instability or excessive temperature."
            )

        return IndividualAssertionResult(
            assertion_type=self.assertion_type,
            assertion_name=self.assertion_name,
            passed=passed,
            message=message,
            score=average_similarity,
            threshold=threshold,
            details={
                "n_samples": n_samples,
                "average_similarity": average_similarity,
                "min_similarity": min_similarity,
                "max_similarity": max_similarity,
                "n_pairs": len(pair_similarities),
                "response_previews": [r[:80] for r in responses],
            },
        )
