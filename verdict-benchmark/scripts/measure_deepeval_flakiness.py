#!/usr/bin/env python3
"""Measure DeepEval flakiness: run AnswerRelevancyMetric N times on the same input.

DeepEval's semantic scoring uses LLM-as-judge internally. Each run makes
an additional API call to score the output, and the LLM judge itself is
stochastic. This script measures the variance in DeepEval's scores across
repeated runs on identical input, using a recorded baseline to avoid
re-calling the model under test.

Each run costs approximately $0.01-0.03 in LLM-as-judge API fees.
20 runs: approximately $0.20-0.60 total.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = BENCHMARK_ROOT / "baselines"
RESULTS_DIR = BENCHMARK_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUNS = 20


def load_baseline_content(model_dir: str, prompt_name: str) -> str | None:
    filepath = BASELINES_DIR / model_dir / f"{prompt_name}.json"
    if not filepath.exists():
        return None
    baseline = json.loads(filepath.read_text())
    return baseline["response_content"]


def load_prompt(name: str) -> str:
    return (BENCHMARK_ROOT / "prompts" / f"{name}.txt").read_text().strip()


def measure() -> dict:
    from deepeval.metrics import AnswerRelevancyMetric
    from deepeval.test_case import LLMTestCase

    prompt_text = load_prompt("semantic_intent")
    # Use a recorded baseline so we do not re-call the model under test.
    # This isolates judge variance from model variance.
    recorded_output = load_baseline_content("gpt4o_2024_11_20", "semantic_intent")
    if recorded_output is None:
        return {"error": "Baseline gpt4o_2024_11_20/semantic_intent not recorded."}

    scores = []
    pass_count = 0
    threshold = 0.7

    print(f"Running {RUNS} iterations of DeepEval AnswerRelevancyMetric...")
    print(f"Each iteration calls an LLM judge. Estimated cost: ${RUNS * 0.02:.2f}")
    print()

    total_start = time.monotonic()

    for i in range(RUNS):
        iteration_start = time.monotonic()

        test_case = LLMTestCase(
            input=prompt_text,
            actual_output=recorded_output,
        )
        metric = AnswerRelevancyMetric(threshold=threshold)
        metric.measure(test_case)

        score = metric.score
        scores.append(score)
        if score >= threshold:
            pass_count += 1

        iteration_ms = int((time.monotonic() - iteration_start) * 1000)
        status = "PASS" if score >= threshold else "FAIL"
        print(f"  [{i+1:2d}/{RUNS}] score={score:.4f} [{status}] ({iteration_ms}ms)")

    total_ms = int((time.monotonic() - total_start) * 1000)

    stdev = round(statistics.stdev(scores), 6) if len(scores) > 1 else 0.0
    score_range = round(max(scores) - min(scores), 6)

    result = {
        "tool": "deepeval",
        "assertion": "AnswerRelevancyMetric",
        "scoring_method": "llm_as_judge",
        "runs": RUNS,
        "pass_count": pass_count,
        "fail_count": RUNS - pass_count,
        "pass_rate": round(pass_count / RUNS, 4),
        "scores_mean": round(statistics.mean(scores), 6),
        "scores_stdev": stdev,
        "scores_min": round(min(scores), 6),
        "scores_max": round(max(scores), 6),
        "score_range": score_range,
        "threshold": threshold,
        "elapsed_ms": total_ms,
        "avg_ms_per_run": round(total_ms / RUNS),
        "note": (
            "DeepEval uses LLM-as-judge for semantic scoring. "
            "Each run makes an additional API call. "
            "Score variance reflects judge nondeterminism."
        ),
    }

    print()
    print(f"Summary:")
    print(f"  Mean score:  {result['scores_mean']:.4f}")
    print(f"  Stdev:       {result['scores_stdev']:.6f}")
    print(f"  Range:       [{result['scores_min']:.4f}, {result['scores_max']:.4f}]")
    print(f"  Pass rate:   {result['pass_rate']:.2%}")
    print(f"  Total time:  {total_ms}ms ({result['avg_ms_per_run']}ms avg)")

    return result


def main() -> None:
    deepeval_result = measure()

    # Merge into existing flakiness.json
    results_path = RESULTS_DIR / "flakiness.json"
    existing = {}
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            pass

    existing["deepeval"] = {"semantic": deepeval_result}
    results_path.write_text(json.dumps(existing, indent=2))
    print(f"\nResults merged into {results_path}")


if __name__ == "__main__":
    main()
