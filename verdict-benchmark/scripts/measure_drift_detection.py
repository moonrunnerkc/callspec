#!/usr/bin/env python3
"""Measure semantic drift detection capability across tools.

Tests whether each tool can detect a 0.15 cosine drift between
recorded baselines from different dated model versions.

Usage:
    python scripts/measure_drift_detection.py --tool verdict
    python scripts/measure_drift_detection.py --tool deepeval
    python scripts/measure_drift_detection.py --all
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = BENCHMARK_ROOT / "baselines"
RESULTS_DIR = BENCHMARK_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = 0.15

OPENAI_PAIRS = [
    ("gpt4o_2024_05_13", "gpt4o_2024_11_20"),
]

ANTHROPIC_PAIRS = [
    ("claude_3_haiku_20240307", "claude_sonnet_4_20250514"),
]

PROMPT_NAMES = [
    "structured_output",
    "semantic_intent",
    "format_compliance",
    "code_generation",
    "numeric_reasoning",
    "instruction_following",
    "chain_of_thought",
]


def load_baseline(model_dir: str, prompt_name: str) -> dict | None:
    filepath = BASELINES_DIR / model_dir / f"{prompt_name}.json"
    if not filepath.exists():
        return None
    return json.loads(filepath.read_text())


def compute_drift_verdict(old_content: str, new_content: str) -> dict:
    """Use Verdict's built-in embedding scorer to compute drift."""
    start = time.monotonic()
    from verdict.scoring.embeddings import score_similarity

    similarity = score_similarity(old_content, new_content, model_name="all-MiniLM-L6-v2")
    elapsed_ms = int((time.monotonic() - start) * 1000)

    drift = 1.0 - similarity
    detected = drift > DRIFT_THRESHOLD

    return {
        "method": "verdict_embedding_scorer",
        "similarity": round(similarity, 6),
        "drift": round(drift, 6),
        "threshold": DRIFT_THRESHOLD,
        "detected": detected,
        "elapsed_ms": elapsed_ms,
        "api_calls": 0,
        "cost_usd": 0.0,
    }


def compute_drift_manual_embeddings(old_content: str, new_content: str) -> dict:
    """Manual embedding comparison (what DeepEval/LangSmith/Braintrust require)."""
    start = time.monotonic()
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([old_content, new_content])
    similarity = float(np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    ))
    elapsed_ms = int((time.monotonic() - start) * 1000)

    drift = 1.0 - similarity
    detected = drift > DRIFT_THRESHOLD

    return {
        "method": "manual_embeddings",
        "similarity": round(similarity, 6),
        "drift": round(drift, 6),
        "threshold": DRIFT_THRESHOLD,
        "detected": detected,
        "elapsed_ms": elapsed_ms,
        "api_calls": 0,
        "cost_usd": 0.0,
        "note": "Not a native feature of DeepEval, LangSmith, or Braintrust. Requires manual code.",
    }


def run_drift_measurements(tool: str) -> list:
    results = []

    all_pairs = OPENAI_PAIRS + ANTHROPIC_PAIRS

    for old_dir, new_dir in all_pairs:
        for prompt_name in PROMPT_NAMES:
            old_baseline = load_baseline(old_dir, prompt_name)
            new_baseline = load_baseline(new_dir, prompt_name)

            if old_baseline is None or new_baseline is None:
                print(f"  Skipping {old_dir} vs {new_dir} / {prompt_name}: baselines not recorded")
                continue

            old_content = old_baseline["response_content"]
            new_content = new_baseline["response_content"]

            pair_label = f"{old_dir}_vs_{new_dir}/{prompt_name}"
            print(f"  Measuring drift: {pair_label}")

            if tool == "verdict":
                measurement = compute_drift_verdict(old_content, new_content)
            else:
                # For non-Verdict tools, we measure what the manual approach yields
                # since none of them have built-in drift detection
                measurement = compute_drift_manual_embeddings(old_content, new_content)

            measurement["pair"] = pair_label
            measurement["old_model"] = old_baseline.get("model_returned", old_dir)
            measurement["new_model"] = new_baseline.get("model_returned", new_dir)
            measurement["old_length"] = len(old_content)
            measurement["new_length"] = len(new_content)

            status = "DETECTED" if measurement["detected"] else "below threshold"
            print(f"    drift={measurement['drift']:.4f} similarity={measurement['similarity']:.4f} [{status}]")

            results.append(measurement)

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Measure drift detection capability.")
    parser.add_argument("--tool", choices=["verdict", "deepeval", "promptfoo", "langsmith", "braintrust"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.tool and not args.all:
        parser.print_help()
        sys.exit(1)

    tools = ["verdict", "deepeval", "promptfoo", "langsmith", "braintrust"] if args.all else [args.tool]
    all_results = {}

    for tool in tools:
        print(f"\n=== Drift Detection: {tool} ===")
        measurements = run_drift_measurements(tool)
        all_results[tool] = {
            "has_native_drift_detection": tool == "verdict",
            "measurements": measurements,
        }

    results_path = RESULTS_DIR / "drift_detection.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
