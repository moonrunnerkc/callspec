#!/usr/bin/env python3
"""Measure flakiness: run the same assertion 100 times and report variance.

This tests the reliability of each tool's semantic assertions by running
them repeatedly against the same input. Tools with statistical confidence
intervals (Verdict) should show lower variance than tools with raw
threshold comparisons (DeepEval, Promptfoo).

Only tests assertions that do not require live API calls (using recorded
baselines) to isolate assertion flakiness from provider variance.

Usage:
    python scripts/measure_flakiness.py --tool verdict --runs 100
    python scripts/measure_flakiness.py --tool deepeval --runs 100
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = BENCHMARK_ROOT / "baselines"
RESULTS_DIR = BENCHMARK_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline_content(model_dir: str, prompt_name: str) -> str | None:
    filepath = BASELINES_DIR / model_dir / f"{prompt_name}.json"
    if not filepath.exists():
        return None
    data = json.loads(filepath.read_text())
    return data["response_content"]


def measure_verdict_flakiness(runs: int) -> dict:
    """Run Verdict's semantic similarity assertion N times on the same input."""
    from verdict.scoring.embeddings import score_similarity

    content_a = load_baseline_content("claude_3_haiku_20240307", "semantic_intent")
    content_b = load_baseline_content("claude_sonnet_4_20250514", "semantic_intent")

    if content_a is None or content_b is None:
        return {"error": "Baselines not recorded. Run record_baselines.py first."}

    # Reference intent for semantic matching
    reference_intent = (
        "An educational explanation of relational database normalization "
        "covering first, second, and third normal forms with a library "
        "book tracking example"
    )

    scores = []
    pass_count = 0
    threshold = 0.70

    print(f"  Running {runs} iterations of semantic_intent_matches...")
    start = time.monotonic()

    for i in range(runs):
        score = score_similarity(content_a, reference_intent, model_name="all-MiniLM-L6-v2")
        scores.append(score)
        if score >= threshold:
            pass_count += 1

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return {
        "tool": "verdict",
        "assertion": "semantic_intent_matches",
        "runs": runs,
        "pass_count": pass_count,
        "fail_count": runs - pass_count,
        "pass_rate": pass_count / runs,
        "scores_mean": round(statistics.mean(scores), 6),
        "scores_stdev": round(statistics.stdev(scores), 6) if runs > 1 else 0.0,
        "scores_min": round(min(scores), 6),
        "scores_max": round(max(scores), 6),
        "threshold": threshold,
        "elapsed_ms": elapsed_ms,
        "note": "Verdict embeddings are deterministic: same input always produces same score. Zero variance expected.",
    }


def measure_structural_flakiness(runs: int) -> dict:
    """Run a structural assertion N times. Should be perfectly deterministic."""
    from verdict.assertions.structural import IsValidJson
    from verdict.core.config import VerdictConfig

    content = load_baseline_content("claude_3_haiku_20240307", "structured_output")
    if content is None:
        return {"error": "Baselines not recorded."}

    assertion = IsValidJson()
    config = VerdictConfig()
    pass_count = 0

    print(f"  Running {runs} iterations of is_valid_json...")
    start = time.monotonic()

    for _ in range(runs):
        result = assertion.evaluate(content, config)
        if result.passed:
            pass_count += 1

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return {
        "tool": "verdict",
        "assertion": "is_valid_json",
        "runs": runs,
        "pass_count": pass_count,
        "fail_count": runs - pass_count,
        "pass_rate": pass_count / runs,
        "elapsed_ms": elapsed_ms,
        "note": "Structural assertions are fully deterministic.",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Measure assertion flakiness.")
    parser.add_argument("--tool", choices=["verdict", "deepeval"], default="verdict")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    results = {}

    if args.tool == "verdict":
        print("=== Verdict Flakiness Measurement ===")
        results["semantic"] = measure_verdict_flakiness(args.runs)
        results["structural"] = measure_structural_flakiness(args.runs)

    elif args.tool == "deepeval":
        print("=== DeepEval Flakiness Measurement ===")
        print("  NOTE: DeepEval semantic assertions use LLM-as-judge.")
        print("  Each run makes an additional API call. Cost: ~$0.01-0.03 per run.")
        print(f"  Total estimated cost for {args.runs} runs: ${args.runs * 0.02:.2f}")
        print("  Skipping automated measurement to avoid unexpected API charges.")
        print("  To run manually: deepeval test run suites/deepeval/test_regression.py")
        results["note"] = (
            f"DeepEval flakiness measurement skipped to avoid ${args.runs * 0.02:.2f} "
            f"in LLM-as-judge API costs. Manual measurement instructions in setup_notes.md."
        )

    results_path = RESULTS_DIR / "flakiness.json"

    # Merge with existing results if present
    existing = {}
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            pass

    existing[args.tool] = results
    results_path.write_text(json.dumps(existing, indent=2))
    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
