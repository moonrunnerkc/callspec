"""LLMAssert pytest integration: regression detection against dated model versions.

This test file demonstrates LLMAssert's pytest plugin for detecting
behavioral drift between model versions. It uses recorded baselines
and live model calls to measure semantic drift.

Run with:
    pytest suites/verdict/test_regression.py -v --tb=short
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINES_DIR = BENCHMARK_ROOT / "baselines"
PROMPTS_DIR = BENCHMARK_ROOT / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text().strip()


def load_baseline(model_dir: str, prompt_name: str) -> dict:
    filepath = BASELINES_DIR / model_dir / f"{prompt_name}.json"
    if not filepath.exists():
        pytest.skip(f"Baseline not recorded: {filepath}")
    return json.loads(filepath.read_text())


def has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


# -- OpenAI drift detection tests --

@pytest.mark.skipif(not has_openai(), reason="OPENAI_API_KEY not set")
class TestOpenAIDrift:
    """Detect behavioral drift between gpt-4o-2024-05-13 and gpt-4o-2024-11-20."""

    def _get_provider(self, model: str):
        from llm_assert.providers.openai import OpenAIProvider
        return OpenAIProvider(model=model, temperature=0.0, seed=42)

    def _run_and_compare(self, prompt_name: str, baseline_model_dir: str, current_model: str):
        """Run prompt against current model and compare to recorded baseline."""
        from llm_assert.scoring.embeddings import score_similarity

        baseline = load_baseline(baseline_model_dir, prompt_name)
        baseline_content = baseline["response_content"]

        provider = self._get_provider(current_model)
        response = provider.call(
            prompt=load_prompt(prompt_name),
            temperature=0,
            seed=42,
        )

        similarity = score_similarity(
            baseline_content,
            response.content,
            model_name="all-MiniLM-L6-v2",
        )
        drift = 1.0 - similarity

        return {
            "baseline_model": baseline["model_returned"],
            "current_model": response.model,
            "similarity": similarity,
            "drift": drift,
            "baseline_length": len(baseline_content),
            "current_length": len(response.content),
        }

    def test_structured_output_drift(self):
        """Detect drift in JSON structured output between model versions."""
        result = self._run_and_compare(
            "structured_output",
            "gpt4o_2024_05_13",
            "gpt-4o-2024-11-20",
        )
        # Drift above 0.15 (similarity below 0.85) indicates regression
        assert result["drift"] <= 0.15, (
            f"Semantic drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline model: {result['baseline_model']}, "
            f"Current model: {result['current_model']}. "
            f"Baseline length: {result['baseline_length']}, "
            f"Current length: {result['current_length']}."
        )

    def test_semantic_intent_drift(self):
        """Detect drift in explanation quality between model versions."""
        result = self._run_and_compare(
            "semantic_intent",
            "gpt4o_2024_05_13",
            "gpt-4o-2024-11-20",
        )
        assert result["drift"] <= 0.15, (
            f"Semantic drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline model: {result['baseline_model']}, "
            f"Current model: {result['current_model']}."
        )

    def test_format_compliance_drift(self):
        """Detect drift in format compliance between model versions."""
        result = self._run_and_compare(
            "format_compliance",
            "gpt4o_2024_05_13",
            "gpt-4o-2024-11-20",
        )
        assert result["drift"] <= 0.15, (
            f"Semantic drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline model: {result['baseline_model']}, "
            f"Current model: {result['current_model']}."
        )

    def test_code_generation_drift(self):
        """Detect drift in code generation style between model versions.

        Code output is a known area of behavioral change between GPT-4o
        dated versions. The Nov 2024 version produces longer, more detailed
        functions than the May 2024 version.
        """
        result = self._run_and_compare(
            "code_generation",
            "gpt4o_2024_05_13",
            "gpt-4o-2024-11-20",
        )
        assert result["drift"] <= 0.15, (
            f"Code generation drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline model: {result['baseline_model']}, "
            f"Current model: {result['current_model']}. "
            f"Baseline length: {result['baseline_length']}, "
            f"Current length: {result['current_length']}."
        )

    def test_numeric_reasoning_drift(self):
        """Detect drift in numeric reasoning behavior between model versions."""
        result = self._run_and_compare(
            "numeric_reasoning",
            "gpt4o_2024_05_13",
            "gpt-4o-2024-11-20",
        )
        assert result["drift"] <= 0.15, (
            f"Numeric reasoning drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline model: {result['baseline_model']}, "
            f"Current model: {result['current_model']}."
        )

    def test_instruction_following_drift(self):
        """Detect drift in instruction-following precision."""
        result = self._run_and_compare(
            "instruction_following",
            "gpt4o_2024_05_13",
            "gpt-4o-2024-11-20",
        )
        assert result["drift"] <= 0.15, (
            f"Instruction following drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline model: {result['baseline_model']}, "
            f"Current model: {result['current_model']}."
        )

    def test_chain_of_thought_drift(self):
        """Detect drift in chain-of-thought reasoning structure."""
        result = self._run_and_compare(
            "chain_of_thought",
            "gpt4o_2024_05_13",
            "gpt-4o-2024-11-20",
        )
        assert result["drift"] <= 0.15, (
            f"Chain of thought drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline model: {result['baseline_model']}, "
            f"Current model: {result['current_model']}."
        )


# -- Anthropic drift detection tests --

@pytest.mark.skipif(not has_anthropic(), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicDrift:
    """Detect behavioral drift between claude-3-haiku-20240307 and claude-sonnet-4-20250514."""

    def _get_provider(self, model: str):
        from llm_assert.providers.anthropic import AnthropicProvider
        return AnthropicProvider(model=model, temperature=0.0)

    def _run_and_compare(self, prompt_name: str, baseline_model_dir: str, current_model: str):
        from llm_assert.scoring.embeddings import score_similarity

        baseline = load_baseline(baseline_model_dir, prompt_name)
        baseline_content = baseline["response_content"]

        provider = self._get_provider(current_model)
        response = provider.call(
            prompt=load_prompt(prompt_name),
            temperature=0,
        )

        similarity = score_similarity(
            baseline_content,
            response.content,
            model_name="all-MiniLM-L6-v2",
        )
        drift = 1.0 - similarity

        return {
            "baseline_model": baseline["model_returned"],
            "current_model": response.model,
            "similarity": similarity,
            "drift": drift,
            "baseline_length": len(baseline_content),
            "current_length": len(response.content),
        }

    def test_structured_output_drift(self):
        result = self._run_and_compare(
            "structured_output",
            "claude_3_haiku_20240307",
            "claude-sonnet-4-20250514",
        )
        # Haiku-to-Sonnet transition produces real behavioral differences;
        # threshold is relaxed because these are architecturally distinct models
        assert result["drift"] <= 0.30, (
            f"Semantic drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline: {result['baseline_model']}, "
            f"Current: {result['current_model']}."
        )

    def test_semantic_intent_drift(self):
        result = self._run_and_compare(
            "semantic_intent",
            "claude_3_haiku_20240307",
            "claude-sonnet-4-20250514",
        )
        assert result["drift"] <= 0.30, (
            f"Semantic drift detected: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline: {result['baseline_model']}, "
            f"Current: {result['current_model']}."
        )

    def test_code_generation_drift(self):
        result = self._run_and_compare(
            "code_generation",
            "claude_3_haiku_20240307",
            "claude-sonnet-4-20250514",
        )
        assert result["drift"] <= 0.30, (
            f"Code generation drift: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline: {result['baseline_model']}, "
            f"Current: {result['current_model']}."
        )

    def test_chain_of_thought_drift(self):
        result = self._run_and_compare(
            "chain_of_thought",
            "claude_3_haiku_20240307",
            "claude-sonnet-4-20250514",
        )
        # Chain-of-thought reasoning differs substantially between model tiers
        assert result["drift"] <= 0.35, (
            f"Chain of thought drift: {result['drift']:.4f} "
            f"(similarity {result['similarity']:.4f}). "
            f"Baseline: {result['baseline_model']}, "
            f"Current: {result['current_model']}."
        )
