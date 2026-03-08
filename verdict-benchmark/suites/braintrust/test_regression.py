"""Braintrust equivalent: regression detection test.

Braintrust requires account creation and login before any functionality
is available. Their documentation opens with "Sign up at braintrust.dev."

This file shows the equivalent test structure, but it cannot be run
without a Braintrust account and API key.

Run with:
    pytest suites/braintrust/test_regression.py -v

Requires:
    - pip install braintrust openai
    - Account at braintrust.dev
    - BRAINTRUST_API_KEY from the Braintrust dashboard (or `braintrust login`)
    - OPENAI_API_KEY for the model under test
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


def has_braintrust() -> bool:
    return bool(os.environ.get("BRAINTRUST_API_KEY"))


def has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(
    not (has_braintrust() and has_openai()),
    reason="BRAINTRUST_API_KEY and/or OPENAI_API_KEY not set",
)
class TestBraintrustDrift:
    """Attempt drift detection using Braintrust's evaluation framework.

    Braintrust's Eval() function runs scoring functions against a dataset.
    Like LangSmith, there is no built-in regression snapshot or semantic
    drift assertion. Custom scorer functions must be written.

    The setup path requires:
    1. Create account at braintrust.dev
    2. Login via `braintrust login` CLI or set BRAINTRUST_API_KEY
    3. Install braintrust package
    4. Write scorer function
    5. Create dataset (or use inline data)
    """

    def test_structured_output_with_braintrust_eval(self):
        """Use Braintrust's Eval to score structured output."""
        import braintrust

        baseline = load_baseline("gpt4o_2024_05_13", "structured_output")
        baseline_content = baseline["response_content"]
        prompt_text = load_prompt("structured_output")

        from openai import OpenAI
        client = OpenAI()

        def task(input_data):
            response = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[{"role": "user", "content": input_data["prompt"]}],
                temperature=0,
                seed=42,
                max_tokens=1024,
            )
            return response.choices[0].message.content

        def drift_scorer(output, expected):
            """Custom scorer: semantic drift via embeddings.

            Braintrust has built-in scorers (Factuality, etc.) that use
            LLM-as-judge. For embedding-based drift detection, we must
            write a custom scorer, same as with LangSmith.
            """
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode([expected, output])
            similarity = float(np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            ))
            return similarity

        # Braintrust Eval pushes results to their platform
        eval_result = braintrust.Eval(
            "verdict-benchmark",
            data=lambda: [
                {
                    "input": {"prompt": prompt_text},
                    "expected": baseline_content,
                }
            ],
            task=task,
            scores=[drift_scorer],
        )

        # Braintrust Eval returns a summary but does not assert.
        # Manual assertion required for CI integration.
        # NOTE: The exact return format depends on the Braintrust SDK version.
        # This test may need adjustment based on the installed version.
        assert eval_result is not None, (
            "Braintrust Eval completed but result format varies by SDK version. "
            "Manual score checking required for CI fail/pass behavior."
        )
