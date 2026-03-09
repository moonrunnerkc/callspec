#!/usr/bin/env python3
"""callspec case study: detecting tool-call drift across model providers.

This script demonstrates callspec's core value proposition:
1. Record a baseline tool-call trajectory from GPT-4o
2. Run the same prompt with Claude Sonnet
3. Show the drift report catching behavioral differences

Run:
    export OPENAI_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python case_study/model_swap_drift.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Load .env if present
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

from callspec.core.config import CallspecConfig
from callspec.core.trajectory import ToolCallTrajectory
from callspec.core.trajectory_builder import TrajectoryBuilder
from callspec.providers.openai import OpenAIProvider
from callspec.providers.anthropic import AnthropicProvider
from callspec.snapshots.manager import SnapshotManager
from callspec.snapshots.diff import SnapshotDiff

# -----------------------------------------------------------------------
# Tool definitions (OpenAI format, converted to Anthropic format below)
# -----------------------------------------------------------------------

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search for available flights between two airports on a given date.",
            "parameters": {
                "type": "object",
                "required": ["origin", "destination", "date"],
                "properties": {
                    "origin": {"type": "string", "description": "IATA code of departure airport"},
                    "destination": {"type": "string", "description": "IATA code of arrival airport"},
                    "date": {"type": "string", "description": "Travel date in YYYY-MM-DD format"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather forecast for a city on a given date.",
            "parameters": {
                "type": "object",
                "required": ["city", "date"],
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a specific flight by its flight ID.",
            "parameters": {
                "type": "object",
                "required": ["flight_id", "passenger_name"],
                "properties": {
                    "flight_id": {"type": "string", "description": "Unique flight identifier"},
                    "passenger_name": {"type": "string", "description": "Full name of the passenger"},
                },
            },
        },
    },
]

TOOLS_ANTHROPIC = [
    {
        "name": t["function"]["name"],
        "description": t["function"]["description"],
        "input_schema": t["function"]["parameters"],
    }
    for t in TOOLS_OPENAI
]

PROMPT = (
    "I want to travel from San Francisco to New York on March 15th 2026. "
    "Help me plan the trip: check the weather at my destination, find flights, "
    "and go ahead and book the best option. My name is Brad Kinnard."
)

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
SNAPSHOT_KEY = "booking_flow"


def print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def print_trajectory(trajectory: ToolCallTrajectory) -> None:
    for i, call in enumerate(trajectory.calls):
        args_short = json.dumps(call.arguments, separators=(",", ":"))
        if len(args_short) > 80:
            args_short = args_short[:77] + "..."
        print(f"  [{i}] {call.tool_name}({args_short})")


def run_openai_baseline() -> ToolCallTrajectory:
    """Call GPT-4o and record the baseline trajectory."""
    print_header("STEP 1: Record baseline with GPT-4o")

    provider = OpenAIProvider(model="gpt-4o", temperature=0, seed=42)
    response = provider.call(PROMPT, tools=TOOLS_OPENAI)

    trajectory = ToolCallTrajectory.from_provider_response(response)

    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_ms}ms")
    print(f"Tool calls: {len(trajectory.calls)}")
    print()
    print_trajectory(trajectory)

    # Save as snapshot baseline
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    manager = SnapshotManager(snapshot_dir=str(SNAPSHOT_DIR))
    manager.update_entry(
        snapshot_key=SNAPSHOT_KEY,
        content=response.content,
        prompt=PROMPT,
        tool_calls=[c.to_dict() for c in trajectory.calls],
        model=response.model,
        provider="openai",
    )
    print(f"\nBaseline saved to {SNAPSHOT_DIR}")

    return trajectory


def run_anthropic_comparison() -> ToolCallTrajectory:
    """Call Claude Sonnet and capture its trajectory."""
    print_header("STEP 2: Run same prompt with Claude Sonnet")

    provider = AnthropicProvider(model="claude-sonnet-4-20250514", temperature=0)
    response = provider.call(PROMPT, tools=TOOLS_ANTHROPIC)

    trajectory = ToolCallTrajectory.from_provider_response(response)

    print(f"Model: {response.model}")
    print(f"Latency: {response.latency_ms}ms")
    print(f"Tool calls: {len(trajectory.calls)}")
    print()
    print_trajectory(trajectory)

    return trajectory


def run_drift_report(
    baseline: ToolCallTrajectory,
    current: ToolCallTrajectory,
) -> None:
    """Compare trajectories and show the drift report."""
    print_header("STEP 3: Drift Report")

    # Structural comparison via SnapshotDiff
    baseline_calls = [c.to_dict() for c in baseline.calls]
    current_calls = [c.to_dict() for c in current.calls]

    diff = SnapshotDiff.compare_trajectories(
        snapshot_key=SNAPSHOT_KEY,
        baseline_calls=baseline_calls,
        current_calls=current_calls,
        baseline_model=baseline.model,
        current_model=current.model,
    )

    print(f"Sequence match:    {diff.sequence_match}")
    print(f"Hash match:        {diff.hash_match}")
    print(f"Model changed:     {diff.model_changed}")
    print()

    if not diff.sequence_match:
        print(f"  Baseline sequence: {diff.baseline_tool_names}")
        print(f"  Current sequence:  {diff.current_tool_names}")
        print()

    if diff.tools_added:
        print(f"  Tools added:   {diff.tools_added}")
    if diff.tools_removed:
        print(f"  Tools removed: {diff.tools_removed}")

    if diff.call_diffs:
        changed_diffs = [cd for cd in diff.call_diffs if cd.status != "unchanged"]
        if changed_diffs:
            print("\n  Per-call diffs:")
            for cd in changed_diffs:
                print(f"    {cd.summary_line()}")
            print()

    print(f"  Summary: {diff.summary()}")
    print()
    print(f"  Full report:\n{diff.detailed_report()}")

    # Run callspec trajectory assertions against the current trajectory
    print_header("STEP 4: callspec assertions (baseline contract)")

    config = CallspecConfig()
    builder = TrajectoryBuilder(trajectory=current, config=config)

    # Assert the current trajectory matches what GPT-4o produced
    result = (
        builder
        .calls_tools_in_order(baseline.tool_names)
        .calls_exactly(baseline.tool_names)
        .run()
    )

    for ar in result.assertions:
        status = "PASS" if ar.passed else "FAIL"
        print(f"  [{status}] {ar.assertion_name}: {ar.message}")

    print()
    if result.passed:
        print("  Result: All assertions PASSED. No drift detected.")
    else:
        print("  Result: DRIFT DETECTED. The models produce different tool-call behavior.")

    return result


def main() -> int:
    # Check API keys
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        return 1
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set")
        return 1

    baseline = run_openai_baseline()
    current = run_anthropic_comparison()
    result = run_drift_report(baseline, current)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
