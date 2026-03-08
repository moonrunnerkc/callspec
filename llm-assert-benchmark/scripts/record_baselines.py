#!/usr/bin/env python3
"""Record baseline responses from dated model versions.

Calls each prompt against each dated model version and saves the raw
response to the appropriate baselines/ directory. These recorded outputs
are the ground truth for all drift detection measurements.

Usage:
    python scripts/record_baselines.py --provider openai
    python scripts/record_baselines.py --provider anthropic
    python scripts/record_baselines.py --all

Requires OPENAI_API_KEY and/or ANTHROPIC_API_KEY in environment.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = BENCHMARK_ROOT / "prompts"
BASELINES_DIR = BENCHMARK_ROOT / "baselines"

# Dated model versions that exhibited real behavioral differences.
# These are the actual endpoints, not aliases.
OPENAI_MODELS = {
    "gpt4o_2024_05_13": "gpt-4o-2024-05-13",
    "gpt4o_2024_11_20": "gpt-4o-2024-11-20",
}

ANTHROPIC_MODELS = {
    "claude_3_haiku_20240307": "claude-3-haiku-20240307",
    "claude_sonnet_4_20250514": "claude-sonnet-4-20250514",
}

PROMPT_FILES = [
    "structured_output.txt",
    "semantic_intent.txt",
    "format_compliance.txt",
    "code_generation.txt",
    "numeric_reasoning.txt",
    "instruction_following.txt",
    "chain_of_thought.txt",
]


def load_prompt(filename: str) -> str:
    filepath = PROMPTS_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Prompt file missing: {filepath}")
    return filepath.read_text().strip()


def record_openai(model_dir_name: str, model_id: str) -> None:
    """Record baselines from an OpenAI dated model version."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    output_dir = BASELINES_DIR / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_file in PROMPT_FILES:
        prompt_name = prompt_file.replace(".txt", "")
        prompt_text = load_prompt(prompt_file)

        print(f"  Recording {model_id} / {prompt_name}...", end=" ", flush=True)
        start = time.monotonic()

        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            seed=42,
            max_tokens=1024,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        content = response.choices[0].message.content or ""
        actual_model = response.model or model_id

        baseline_entry = {
            "model_requested": model_id,
            "model_returned": actual_model,
            "provider": "openai",
            "prompt_file": prompt_file,
            "prompt_text": prompt_text,
            "response_content": content,
            "latency_ms": elapsed_ms,
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
            "finish_reason": response.choices[0].finish_reason,
            "temperature": 0,
            "seed": 42,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        output_path = output_dir / f"{prompt_name}.json"
        output_path.write_text(json.dumps(baseline_entry, indent=2))
        print(f"done ({elapsed_ms}ms, {len(content)} chars)")


def record_anthropic(model_dir_name: str, model_id: str) -> None:
    """Record baselines from an Anthropic dated model version."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Error: anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in environment.")
        sys.exit(1)

    client = Anthropic(api_key=api_key)
    output_dir = BASELINES_DIR / model_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for prompt_file in PROMPT_FILES:
        prompt_name = prompt_file.replace(".txt", "")
        prompt_text = load_prompt(prompt_file)

        print(f"  Recording {model_id} / {prompt_name}...", end=" ", flush=True)
        start = time.monotonic()

        response = client.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0,
            max_tokens=1024,
        )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        content = response.content[0].text if response.content else ""
        actual_model = response.model or model_id

        baseline_entry = {
            "model_requested": model_id,
            "model_returned": actual_model,
            "provider": "anthropic",
            "prompt_file": prompt_file,
            "prompt_text": prompt_text,
            "response_content": content,
            "latency_ms": elapsed_ms,
            "prompt_tokens": response.usage.input_tokens if response.usage else None,
            "completion_tokens": response.usage.output_tokens if response.usage else None,
            "finish_reason": response.stop_reason,
            "temperature": 0,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        output_path = output_dir / f"{prompt_name}.json"
        output_path.write_text(json.dumps(baseline_entry, indent=2))
        print(f"done ({elapsed_ms}ms, {len(content)} chars)")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Record baseline responses from dated model versions."
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="Record baselines for a specific provider.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Record baselines for all providers with available API keys.",
    )
    args = parser.parse_args()

    if not args.provider and not args.all:
        parser.print_help()
        sys.exit(1)

    if args.provider == "openai" or args.all:
        if os.environ.get("OPENAI_API_KEY"):
            print("=== Recording OpenAI baselines ===")
            for dir_name, model_id in OPENAI_MODELS.items():
                print(f"\nModel: {model_id}")
                record_openai(dir_name, model_id)
        elif args.provider == "openai":
            print("Error: OPENAI_API_KEY not set.")
            sys.exit(1)
        else:
            print("Skipping OpenAI (no OPENAI_API_KEY in environment)")

    if args.provider == "anthropic" or args.all:
        if os.environ.get("ANTHROPIC_API_KEY"):
            print("\n=== Recording Anthropic baselines ===")
            for dir_name, model_id in ANTHROPIC_MODELS.items():
                print(f"\nModel: {model_id}")
                record_anthropic(dir_name, model_id)
        elif args.provider == "anthropic":
            print("Error: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        else:
            print("Skipping Anthropic (no ANTHROPIC_API_KEY in environment)")

    print("\nBaseline recording complete.")


if __name__ == "__main__":
    main()
