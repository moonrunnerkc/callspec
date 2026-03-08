"""DeepEval equivalent: regression detection test.

This file implements the same drift detection test using DeepEval's
API. It demonstrates the setup overhead, the LLM-as-judge default,
and the lack of built-in regression snapshot capability.

DeepEval does not have a native regression/baseline comparison feature.
To approximate it, we manually load the baseline, call the model, and
use DeepEval's AnswerRelevancyMetric (which internally uses LLM-as-judge)
to score the output. Semantic drift must be computed manually since
DeepEval does not provide embedding-based similarity scoring by default.

Run with:
    deepeval test run suites/deepeval/test_regression.py
    (or: pytest suites/deepeval/test_regression.py -v)

Requires: pip install deepeval openai
Requires: OPENAI_API_KEY in environment (both for the model under test
          AND for DeepEval's LLM-as-judge scoring)
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


@pytest.mark.skipif(not has_openai(), reason="OPENAI_API_KEY not set")
class TestDeepEvalDrift:
    """Attempt drift detection using DeepEval.

    DeepEval's approach: use AnswerRelevancyMetric (LLM-as-judge) to score
    the current output. There is no built-in mechanism to compare against
    a recorded baseline using embedding similarity. The LLM-as-judge call
    is an additional API call per assertion, adding cost.
    """

    def test_structured_output_relevancy(self):
        """Score current model output using DeepEval's LLM-as-judge."""
        from deepeval import assert_test
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        from openai import OpenAI

        prompt_text = load_prompt("structured_output")

        # Call the model under test (same as LLMAssert does)
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            seed=42,
            max_tokens=1024,
        )
        actual_output = response.choices[0].message.content

        # DeepEval test case (no baseline comparison, just relevancy scoring)
        # NOTE: This metric invokes an additional LLM call for judging
        test_case = LLMTestCase(
            input=prompt_text,
            actual_output=actual_output,
        )
        metric = AnswerRelevancyMetric(threshold=0.7)

        # assert_test runs the LLM judge and raises on failure
        assert_test(test_case, [metric])

    def test_semantic_intent_relevancy(self):
        """Score semantic intent output using DeepEval's LLM-as-judge."""
        from deepeval import assert_test
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase
        from openai import OpenAI

        prompt_text = load_prompt("semantic_intent")

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            seed=42,
            max_tokens=1024,
        )
        actual_output = response.choices[0].message.content

        test_case = LLMTestCase(
            input=prompt_text,
            actual_output=actual_output,
        )
        metric = AnswerRelevancyMetric(threshold=0.7)
        assert_test(test_case, [metric])

    def test_manual_drift_comparison(self):
        """Manual drift detection: DeepEval has no built-in baseline comparison.

        We compute cosine similarity ourselves using sentence-transformers,
        which is the same approach LLMAssert uses natively. This test exists
        to show that detecting drift in DeepEval requires manual embedding
        computation that is not part of DeepEval's API.
        """
        from openai import OpenAI
        from sentence_transformers import SentenceTransformer

        baseline = load_baseline("gpt4o_2024_05_13", "structured_output")
        baseline_content = baseline["response_content"]

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[{"role": "user", "content": load_prompt("structured_output")}],
            temperature=0,
            seed=42,
            max_tokens=1024,
        )
        current_content = response.choices[0].message.content

        # Manual embedding comparison (not a DeepEval feature)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode([baseline_content, current_content])
        import numpy as np
        similarity = float(np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        ))
        drift = 1.0 - similarity

        assert drift <= 0.15, (
            f"Drift {drift:.4f} exceeds threshold. "
            f"This drift detection required manual embedding code "
            f"outside of DeepEval's API."
        )
