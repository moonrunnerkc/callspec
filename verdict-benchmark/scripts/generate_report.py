#!/usr/bin/env python3
"""Generate a comparison report from all measurement results.

Reads results JSON files and produces a formatted summary table
for the benchmark README.

Usage:
    python scripts/generate_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

TOOLS = ["verdict", "deepeval", "promptfoo", "langsmith", "braintrust"]


def load_json(filename: str) -> dict:
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        return {}
    try:
        return json.loads(filepath.read_text())
    except json.JSONDecodeError:
        return {}


def format_table() -> str:
    setup = load_json("setup_time.json")
    drift = load_json("drift_detection.json")
    cost = load_json("cost.json")
    exit_codes = load_json("exit_codes.json")
    flakiness = load_json("flakiness.json")
    loc = load_json("loc.json")
    api_calls = load_json("api_call_counts.json")

    lines = []
    lines.append("# Verdict Benchmark Results")
    lines.append("")
    lines.append("| Metric | Verdict | DeepEval | Promptfoo | LangSmith | Braintrust |")
    lines.append("|--------|---------|----------|-----------|-----------|------------|")

    # Account required
    lines.append("| Account required | No | Partial | No | Yes | Yes |")

    # Language
    lines.append("| Language | Python | Python | Node.js | Python | Python |")

    # Setup time (tiered)
    # Verdict has three tiers: base, with_provider, full (semantic)
    # Use base for the primary comparison; show full separately
    verdict_base = setup.get("verdict_base", setup.get("verdict", {}))
    verdict_full = setup.get("verdict_full", {})
    deepeval_setup = setup.get("deepeval", {})

    row = "| Install time (base) |"
    if verdict_base.get("install_ms"):
        row += f" {verdict_base['install_ms'] / 1000:.1f}s |"
    else:
        row += " -- |"
    if deepeval_setup.get("install_ms"):
        row += f" {deepeval_setup['install_ms'] / 1000:.1f}s |"
    else:
        row += " -- |"
    row += " -- | N/A* | N/A* |"
    lines.append(row)

    row = "| Install time (full semantic) |"
    if verdict_full.get("install_ms"):
        row += f" {verdict_full['install_ms'] / 1000:.1f}s |"
    else:
        row += " -- |"
    if deepeval_setup.get("install_ms"):
        row += f" {deepeval_setup['install_ms'] / 1000:.1f}s |"
    else:
        row += " -- |"
    row += " -- | N/A* | N/A* |"
    lines.append(row)

    row = "| First test execution |"
    if verdict_base.get("first_test_ms"):
        row += f" {verdict_base['first_test_ms']}ms |"
    else:
        row += " -- |"
    if deepeval_setup.get("first_test_ms"):
        row += f" {deepeval_setup['first_test_ms']}ms |"
    else:
        row += " -- |"
    row += " -- | N/A* | N/A* |"
    lines.append(row)

    row = "| Total setup-to-first-test (base) |"
    if verdict_base.get("warm_ms"):
        row += f" {verdict_base['warm_ms'] / 1000:.1f}s |"
    else:
        row += " -- |"
    if deepeval_setup.get("warm_ms"):
        row += f" {deepeval_setup['warm_ms'] / 1000:.1f}s |"
    else:
        row += " -- |"
    row += " -- | N/A* | N/A* |"
    lines.append(row)

    # Drift detection
    row = "| Catches 0.15 drift |"
    for tool in TOOLS:
        if tool in drift:
            has_native = drift[tool].get("has_native_drift_detection", False)
            measurements = drift[tool].get("measurements", [])
            any_detected = any(m.get("detected", False) for m in measurements)
            if has_native and any_detected:
                row += " Yes (native) |"
            elif any_detected:
                row += " Manual only |"
            else:
                row += " No |"
        else:
            row += " -- |"
    lines.append(row)

    # Flakiness (Verdict)
    row = "| Flakiness stdev (semantic) |"
    for tool in TOOLS:
        if tool in flakiness:
            tool_data = flakiness[tool]
            if "semantic" in tool_data and "scores_stdev" in tool_data["semantic"]:
                stdev = tool_data["semantic"]["scores_stdev"]
                row += f" {stdev} |"
            elif "note" in tool_data:
                row += " See notes |"
            else:
                row += " -- |"
        else:
            row += " -- |"
    lines.append(row)

    # Flakiness score range
    row = "| Flakiness score range |"
    for tool in TOOLS:
        if tool in flakiness:
            tool_data = flakiness[tool]
            if "semantic" in tool_data and "score_range" in tool_data["semantic"]:
                score_range = tool_data["semantic"]["score_range"]
                row += f" {score_range} |"
            elif "semantic" in tool_data:
                stdev = tool_data["semantic"].get("scores_stdev", 0)
                row += f" {0.0 if stdev == 0 else '--'} |"
            else:
                row += " -- |"
        else:
            row += " -- |"
    lines.append(row)

    # Exit codes
    row = "| Exit code on failure (CLI) |"
    for tool in TOOLS:
        if tool in exit_codes:
            code = exit_codes[tool].get("cli", exit_codes[tool].get("deepeval_cli", "--"))
            row += f" {code} |"
        else:
            row += " -- |"
    lines.append(row)

    row = "| Exit code on failure (pytest) |"
    for tool in TOOLS:
        if tool in exit_codes and "pytest" in exit_codes[tool]:
            row += f" {exit_codes[tool]['pytest']} |"
        else:
            row += " -- |"
    lines.append(row)

    # Measured API calls
    row = "| Measured provider API calls |"
    for tool in TOOLS:
        if tool in api_calls and "provider_api_calls" in api_calls[tool]:
            row += f" {api_calls[tool]['provider_api_calls']} |"
        else:
            row += " -- |"
    lines.append(row)

    # Telemetry calls
    row = "| Telemetry calls (uninvited) |"
    for tool in TOOLS:
        if tool in api_calls and "calls_by_host" in api_calls[tool]:
            telemetry_hosts = {"us.i.posthog.com", "api.ipify.org", "posthog.com"}
            telemetry_count = sum(
                count for host, count in api_calls[tool]["calls_by_host"].items()
                if host in telemetry_hosts
            )
            row += f" {telemetry_count} |"
        else:
            row += " -- |"
    lines.append(row)

    # LOC for drift detection
    row = "| Lines of code (drift test) |"
    for tool in TOOLS:
        if tool in loc:
            row += f" {loc[tool]['logic_lines']} |"
        else:
            row += " -- |"
    lines.append(row)

    # Cost
    row = "| API calls per run (estimated) |"
    for tool in TOOLS:
        if tool in cost:
            row += f" {cost[tool]['total_api_calls_per_run']} |"
        else:
            row += " -- |"
    lines.append(row)

    row = "| Monthly CI cost (30/day) |"
    for tool in TOOLS:
        if tool in cost:
            row += f" ${cost[tool]['monthly_cost_usd']} |"
        else:
            row += " -- |"
    lines.append(row)

    # GPT-4o drift details
    lines.append("")
    lines.append("## GPT-4o Version Drift (gpt-4o-2024-05-13 vs gpt-4o-2024-11-20)")
    lines.append("")
    lines.append("| Prompt | Cosine Drift | Detected (>0.15) |")
    lines.append("|--------|-------------|-----------------|")

    if "verdict" in drift:
        for m in drift["verdict"].get("measurements", []):
            if "gpt4o" in m.get("pair", ""):
                prompt_name = m["pair"].split("/")[-1]
                detected = "Yes" if m["detected"] else "No"
                lines.append(f"| {prompt_name} | {m['drift']:.4f} | {detected} |")

    # Anthropic migration drift details
    lines.append("")
    lines.append("## Anthropic Migration Drift (claude-3-haiku vs claude-sonnet-4)")
    lines.append("")
    lines.append("| Prompt | Cosine Drift | Detected (>0.15) |")
    lines.append("|--------|-------------|-----------------|")

    if "verdict" in drift:
        for m in drift["verdict"].get("measurements", []):
            if "claude" in m.get("pair", ""):
                prompt_name = m["pair"].split("/")[-1]
                detected = "Yes" if m["detected"] else "No"
                lines.append(f"| {prompt_name} | {m['drift']:.4f} | {detected} |")

    lines.append("")
    lines.append("*LangSmith and Braintrust require account creation, which cannot be automated.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    report = format_table()
    print(report)

    report_path = RESULTS_DIR / "comparison_report.md"
    report_path.write_text(report)
    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
