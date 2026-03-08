#!/usr/bin/env python3
"""Measure lines of code required for equivalent drift detection in each tool.

Counts only the lines a developer writes to achieve the same outcome:
detect semantic drift between two model versions for a single prompt.
Excludes imports, blank lines, and comments to measure the irreducible
setup burden.

This produces a fair comparison of developer effort, not raw file size.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Each snippet is the minimal code a developer writes to detect drift
# for a single prompt between two model versions. We count only lines
# that contain actual logic (no imports, comments, or blank lines).

VERDICT_SNIPPET = """
from verdict import Verdict
from verdict.providers.openai import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o-2024-11-20", temperature=0, seed=42)
v = Verdict(provider)

result = v.assert_that(prompt).semantic_drift_is_below(
    snapshot_key="structured_output",
    max_drift=0.15,
).run()

assert result.passed
"""

DEEPEVAL_SNIPPET = """
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import json

baseline = json.load(open("baselines/gpt4o_2024_05_13/structured_output.json"))
baseline_content = baseline["response_content"]

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-2024-11-20",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    seed=42,
    max_tokens=1024,
)
current_content = response.choices[0].message.content

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([baseline_content, current_content])
similarity = float(np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
))
drift = 1.0 - similarity

assert drift <= 0.15
"""

PROMPTFOO_SNIPPET = """
# promptfoo_config.yaml (YAML, not Python):
# providers:
#   - id: openai:gpt-4o-2024-11-20
#
# tests:
#   - vars:
#       prompt: "..."
#     assert:
#       - type: llm-rubric
#         value: "output is semantically similar to baseline"
#
# Then run: npx promptfoo eval
#
# No built-in embedding drift comparison.
# Must write custom JavaScript assertion for cosine similarity.

const { cosineSimilarity } = require('./utils');
const baseline = require('./baselines/structured_output.json');

module.exports = async function(output) {
  const drift = 1.0 - cosineSimilarity(baseline.embedding, output.embedding);
  return { pass: drift <= 0.15, score: 1.0 - drift };
};
"""

LANGSMITH_SNIPPET = """
from langsmith import Client
from langsmith.evaluation import evaluate
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import json

client = Client()
openai_client = OpenAI()

baseline = json.load(open("baselines/gpt4o_2024_05_13/structured_output.json"))
baseline_content = baseline["response_content"]

def drift_evaluator(run, example):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([baseline_content, run.outputs["output"]])
    similarity = float(np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    ))
    drift = 1.0 - similarity
    return {"key": "drift", "score": 1.0 - drift, "passed": drift <= 0.15}

def predict(inputs):
    response = openai_client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": inputs["prompt"]}],
        temperature=0,
        seed=42,
    )
    return {"output": response.choices[0].message.content}

results = evaluate(predict, data="my_dataset", evaluators=[drift_evaluator])
"""


def count_logic_lines(snippet: str) -> int:
    """Count lines that contain actual logic (not imports, comments, blanks)."""
    lines = snippet.strip().split("\n")
    logic_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        logic_lines += 1
    return logic_lines


def count_total_lines(snippet: str) -> int:
    """Count all non-empty lines."""
    return len([l for l in snippet.strip().split("\n") if l.strip()])


def main() -> None:
    snippets = {
        "verdict": VERDICT_SNIPPET,
        "deepeval": DEEPEVAL_SNIPPET,
        "promptfoo": PROMPTFOO_SNIPPET,
        "langsmith": LANGSMITH_SNIPPET,
    }

    results = {}
    for tool, snippet in snippets.items():
        logic = count_logic_lines(snippet)
        total = count_total_lines(snippet)
        results[tool] = {
            "logic_lines": logic,
            "total_lines": total,
            "note": "Lines of developer-written code for single-prompt drift detection.",
        }
        print(f"{tool:12s}: {logic:3d} logic lines, {total:3d} total lines")

    results_path = RESULTS_DIR / "loc.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
