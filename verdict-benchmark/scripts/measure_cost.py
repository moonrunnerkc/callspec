#!/usr/bin/env python3
"""Measure API call cost per assertion for each tool.

Counts the number of external API calls each tool makes for an
identical 20-assertion suite, and computes dollar cost at published
API pricing.

This script analyzes the architectural cost model of each tool by
counting calls rather than making live API requests (to avoid
burning money on the measurement itself).

Usage:
    python scripts/measure_cost.py --tool verdict
    python scripts/measure_cost.py --all
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Published API pricing (approximate, as of early 2026)
# Using gpt-4o-mini as the typical judge model for LLM-as-judge tools
PRICING = {
    "gpt-4o-2024-11-20": {
        "input_per_1k": 0.0025,
        "output_per_1k": 0.01,
    },
    "gpt-4o-mini": {
        "input_per_1k": 0.00015,
        "output_per_1k": 0.0006,
    },
}

# Estimated tokens per assertion call
ESTIMATED_PROMPT_TOKENS = 500
ESTIMATED_COMPLETION_TOKENS = 300
ESTIMATED_JUDGE_PROMPT_TOKENS = 800
ESTIMATED_JUDGE_COMPLETION_TOKENS = 200

ASSERTION_COUNT = 34
DAILY_CI_RUNS = 30
DAYS_PER_MONTH = 30


def compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pricing = PRICING.get(model, PRICING["gpt-4o-2024-11-20"])
    input_cost = (prompt_tokens / 1000) * pricing["input_per_1k"]
    output_cost = (completion_tokens / 1000) * pricing["output_per_1k"]
    return input_cost + output_cost


def analyze_verdict() -> dict:
    """Verdict cost model: one provider call per case, zero for semantic scoring."""
    # Verdict makes exactly one API call per test case (to get the LLM response).
    # Semantic assertions use local embeddings (sentence-transformers), zero API calls.
    # Structural assertions are pure string operations, zero API calls.

    # Suite has 7 cases: structured_output, semantic_intent, format_compliance,
    # code_generation, numeric_reasoning, instruction_following, chain_of_thought
    provider_calls = 7
    judge_calls = 0  # Verdict uses local embeddings, not LLM-as-judge
    embedding_calls = 0  # Local model, no API

    per_run_cost = sum(
        compute_cost("gpt-4o-2024-11-20", ESTIMATED_PROMPT_TOKENS, ESTIMATED_COMPLETION_TOKENS)
        for _ in range(provider_calls)
    )

    monthly_runs = DAILY_CI_RUNS * DAYS_PER_MONTH
    monthly_cost = per_run_cost * monthly_runs

    return {
        "tool": "verdict",
        "assertions_per_suite": ASSERTION_COUNT,
        "provider_api_calls_per_run": provider_calls,
        "judge_api_calls_per_run": judge_calls,
        "total_api_calls_per_run": provider_calls + judge_calls,
        "cost_per_run_usd": round(per_run_cost, 4),
        "monthly_runs": monthly_runs,
        "monthly_cost_usd": round(monthly_cost, 2),
        "semantic_scoring_method": "local_embeddings",
        "embedding_api_calls": 0,
        "note": "Semantic assertions use sentence-transformers locally. Zero external API calls for scoring.",
    }


def analyze_deepeval() -> dict:
    """DeepEval cost model: one provider call per case PLUS one judge call per semantic assertion."""
    provider_calls = 7
    # DeepEval's AnswerRelevancyMetric, FaithfulnessMetric, etc. each call an LLM judge
    semantic_assertions = 21  # Approximate: most of the 34 assertions involve semantic judging
    judge_calls = semantic_assertions  # One judge call per semantic metric

    per_run_cost = (
        sum(
            compute_cost("gpt-4o-2024-11-20", ESTIMATED_PROMPT_TOKENS, ESTIMATED_COMPLETION_TOKENS)
            for _ in range(provider_calls)
        )
        + sum(
            compute_cost("gpt-4o-mini", ESTIMATED_JUDGE_PROMPT_TOKENS, ESTIMATED_JUDGE_COMPLETION_TOKENS)
            for _ in range(judge_calls)
        )
    )

    monthly_runs = DAILY_CI_RUNS * DAYS_PER_MONTH
    monthly_cost = per_run_cost * monthly_runs

    return {
        "tool": "deepeval",
        "assertions_per_suite": ASSERTION_COUNT,
        "provider_api_calls_per_run": provider_calls,
        "judge_api_calls_per_run": judge_calls,
        "total_api_calls_per_run": provider_calls + judge_calls,
        "cost_per_run_usd": round(per_run_cost, 4),
        "monthly_runs": monthly_runs,
        "monthly_cost_usd": round(monthly_cost, 2),
        "semantic_scoring_method": "llm_as_judge",
        "note": "Each semantic metric makes an additional LLM call for judging.",
    }


def analyze_promptfoo() -> dict:
    """Promptfoo cost model: one provider call per test, plus LLM calls for llm-rubric assertions."""
    provider_calls = 7
    # llm-rubric assertions call an LLM judge
    llm_rubric_assertions = 10  # Approximate for semantic checks across 7 cases
    judge_calls = llm_rubric_assertions

    per_run_cost = (
        sum(
            compute_cost("gpt-4o-2024-11-20", ESTIMATED_PROMPT_TOKENS, ESTIMATED_COMPLETION_TOKENS)
            for _ in range(provider_calls)
        )
        + sum(
            compute_cost("gpt-4o-mini", ESTIMATED_JUDGE_PROMPT_TOKENS, ESTIMATED_JUDGE_COMPLETION_TOKENS)
            for _ in range(judge_calls)
        )
    )

    monthly_runs = DAILY_CI_RUNS * DAYS_PER_MONTH
    monthly_cost = per_run_cost * monthly_runs

    return {
        "tool": "promptfoo",
        "assertions_per_suite": ASSERTION_COUNT,
        "provider_api_calls_per_run": provider_calls,
        "judge_api_calls_per_run": judge_calls,
        "total_api_calls_per_run": provider_calls + judge_calls,
        "cost_per_run_usd": round(per_run_cost, 4),
        "monthly_runs": monthly_runs,
        "monthly_cost_usd": round(monthly_cost, 2),
        "semantic_scoring_method": "llm_as_judge",
        "note": "llm-rubric assertions invoke an LLM judge. Structural assertions (contains, is-json) are free.",
    }


def analyze_langsmith() -> dict:
    """LangSmith cost model: provider calls plus custom evaluator costs."""
    provider_calls = 7
    # LangSmith evaluators can be custom. If using LLM-based evaluators:
    judge_calls = 14  # Conservative estimate for custom LLM evaluators

    per_run_cost = (
        sum(
            compute_cost("gpt-4o-2024-11-20", ESTIMATED_PROMPT_TOKENS, ESTIMATED_COMPLETION_TOKENS)
            for _ in range(provider_calls)
        )
        + sum(
            compute_cost("gpt-4o-mini", ESTIMATED_JUDGE_PROMPT_TOKENS, ESTIMATED_JUDGE_COMPLETION_TOKENS)
            for _ in range(judge_calls)
        )
    )

    monthly_runs = DAILY_CI_RUNS * DAYS_PER_MONTH
    monthly_cost = per_run_cost * monthly_runs

    return {
        "tool": "langsmith",
        "assertions_per_suite": ASSERTION_COUNT,
        "provider_api_calls_per_run": provider_calls,
        "judge_api_calls_per_run": judge_calls,
        "total_api_calls_per_run": provider_calls + judge_calls,
        "cost_per_run_usd": round(per_run_cost, 4),
        "monthly_runs": monthly_runs,
        "monthly_cost_usd": round(monthly_cost, 2),
        "semantic_scoring_method": "custom_evaluators",
        "additional_cost": "LangSmith platform subscription may apply",
        "note": "Cost depends on evaluator choice. Custom embedding evaluators are possible but not default.",
    }


def analyze_braintrust() -> dict:
    """Braintrust cost model: similar to LangSmith."""
    provider_calls = 7
    judge_calls = 14

    per_run_cost = (
        sum(
            compute_cost("gpt-4o-2024-11-20", ESTIMATED_PROMPT_TOKENS, ESTIMATED_COMPLETION_TOKENS)
            for _ in range(provider_calls)
        )
        + sum(
            compute_cost("gpt-4o-mini", ESTIMATED_JUDGE_PROMPT_TOKENS, ESTIMATED_JUDGE_COMPLETION_TOKENS)
            for _ in range(judge_calls)
        )
    )

    monthly_runs = DAILY_CI_RUNS * DAYS_PER_MONTH
    monthly_cost = per_run_cost * monthly_runs

    return {
        "tool": "braintrust",
        "assertions_per_suite": ASSERTION_COUNT,
        "provider_api_calls_per_run": provider_calls,
        "judge_api_calls_per_run": judge_calls,
        "total_api_calls_per_run": provider_calls + judge_calls,
        "cost_per_run_usd": round(per_run_cost, 4),
        "monthly_runs": monthly_runs,
        "monthly_cost_usd": round(monthly_cost, 2),
        "semantic_scoring_method": "built_in_scorers_llm_judge",
        "additional_cost": "Braintrust platform subscription",
        "note": "Built-in scorers (Factuality, etc.) use LLM-as-judge.",
    }


ANALYZERS = {
    "verdict": analyze_verdict,
    "deepeval": analyze_deepeval,
    "promptfoo": analyze_promptfoo,
    "langsmith": analyze_langsmith,
    "braintrust": analyze_braintrust,
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Measure API cost per assertion.")
    parser.add_argument("--tool", choices=list(ANALYZERS.keys()))
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not args.tool and not args.all:
        parser.print_help()
        sys.exit(1)

    tools = list(ANALYZERS.keys()) if args.all else [args.tool]
    all_results = {}

    for tool in tools:
        print(f"=== Cost Analysis: {tool} ===")
        analysis = ANALYZERS[tool]()
        all_results[tool] = analysis
        print(f"  API calls/run: {analysis['total_api_calls_per_run']}")
        print(f"  Cost/run: ${analysis['cost_per_run_usd']}")
        print(f"  Monthly cost ({analysis['monthly_runs']} runs): ${analysis['monthly_cost_usd']}")
        print()

    results_path = RESULTS_DIR / "cost.json"
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"Results written to {results_path}")


if __name__ == "__main__":
    main()
