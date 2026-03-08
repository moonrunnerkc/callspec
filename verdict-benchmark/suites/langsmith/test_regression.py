"""LangSmith equivalent: regression detection test.

This file demonstrates what the same drift detection test looks like
when using LangSmith. The critical difference: LangSmith requires
account creation and an API key from smith.langchain.com before any
of this code can run.

LangSmith is an observability platform, not an assertion library.
Its primary design is trace-and-observe, not test-and-fail. The
evaluator framework exists but is secondary to the tracing workflow.

Run with:
    pytest suites/langsmith/test_regression.py -v

Requires:
    - pip install langsmith langchain-openai
    - Account at smith.langchain.com
    - LANGCHAIN_API_KEY from the LangSmith dashboard
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


def has_langsmith() -> bool:
    return bool(os.environ.get("LANGCHAIN_API_KEY"))


def has_openai() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(
    not (has_langsmith() and has_openai()),
    reason="LANGCHAIN_API_KEY and/or OPENAI_API_KEY not set",
)
class TestLangSmithDrift:
    """Attempt drift detection using LangSmith's evaluation framework.

    LangSmith's evaluate() runs an evaluator function against a dataset.
    There is no built-in regression snapshot or semantic drift assertion.
    We must implement drift detection manually, then push results to
    the LangSmith platform for visualization.

    The setup path requires:
    1. Create account at smith.langchain.com
    2. Generate API key from dashboard
    3. Set LANGCHAIN_API_KEY environment variable
    4. Install langsmith + langchain-openai packages
    5. Write evaluator function
    6. Create dataset (or use inline examples)
    """

    def test_structured_output_with_langsmith_eval(self):
        """Use LangSmith's evaluator pattern to check structured output."""
        from langchain_openai import ChatOpenAI
        from langsmith import Client
        from langsmith.evaluation import evaluate

        baseline = load_baseline("gpt4o_2024_05_13", "structured_output")
        baseline_content = baseline["response_content"]
        prompt_text = load_prompt("structured_output")

        llm = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0, seed=42)

        def target(inputs: dict) -> dict:
            response = llm.invoke(inputs["prompt"])
            return {"output": response.content}

        def drift_evaluator(run, example) -> dict:
            """Custom evaluator to check semantic drift.

            This is the manual equivalent of Verdict's semantic_drift_is_below().
            LangSmith has no built-in version.
            """
            from sentence_transformers import SentenceTransformer
            import numpy as np

            current = run.outputs["output"]
            reference = example.outputs["baseline"]

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode([reference, current])
            similarity = float(np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            ))

            return {
                "key": "semantic_drift",
                "score": similarity,
                "comment": f"Cosine similarity: {similarity:.4f}",
            }

        # LangSmith requires a dataset. For inline use, create one:
        client = Client()
        dataset_name = "verdict-benchmark-structured-output"

        # Create dataset if it does not already exist
        try:
            dataset = client.create_dataset(dataset_name)
            client.create_example(
                inputs={"prompt": prompt_text},
                outputs={"baseline": baseline_content},
                dataset_id=dataset.id,
            )
        except Exception:
            # Dataset may already exist from a previous run
            datasets = list(client.list_datasets(dataset_name=dataset_name))
            if datasets:
                dataset = datasets[0]
            else:
                raise

        results = evaluate(
            target,
            data=dataset_name,
            evaluators=[drift_evaluator],
        )

        # LangSmith evaluate() returns results but does not assert.
        # We must check the score manually.
        for result in results:
            for eval_result in result.get("evaluation_results", {}).get("results", []):
                if eval_result.get("key") == "semantic_drift":
                    similarity = eval_result["score"]
                    drift = 1.0 - similarity
                    assert drift <= 0.15, (
                        f"Semantic drift {drift:.4f} exceeds threshold. "
                        f"This required: LangSmith account, custom evaluator, "
                        f"dataset creation, and manual assertion logic."
                    )
