"""ReportFormatter: JSON, plaintext, Rich, and JUnit XML output for LLMAssert results.

Formats AssertionResult and SuiteResult into structured output for
terminal display, CI integration, and verdict.run ingestion.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from llm_assert.core.types import AssertionResult, SuiteResult
from llm_assert.version import __version__


class ReportFormatter:
    """Converts LLMAssert results into human-readable and machine-readable formats."""

    @staticmethod
    def to_json(
        suite_result: SuiteResult | None = None,
        assertion_results: dict[str, AssertionResult] | None = None,
        suite_name: str = "default",
    ) -> str:
        """Serialize results to JSON for verdict.run ingestion or file storage."""
        report: dict[str, Any] = {
            "llm_assert_version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "suite_name": suite_name,
        }

        if suite_result:
            report["suite"] = {
                "passed": suite_result.passed,
                "total_cases": suite_result.total_cases,
                "passed_cases": suite_result.passed_cases,
                "failed_cases": suite_result.failed_cases,
                "warned_cases": suite_result.warned_cases,
                "execution_time_ms": suite_result.execution_time_ms,
                "cases": {},
            }
            for case_name, case_result in suite_result.case_results.items():
                report["suite"]["cases"][case_name] = _serialize_assertion_result(case_result)

        if assertion_results:
            report["results"] = {}
            for name, result in assertion_results.items():
                report["results"][name] = _serialize_assertion_result(result)

        return json.dumps(report, indent=2)

    @staticmethod
    def to_plaintext(
        suite_result: SuiteResult | None = None,
        assertion_results: dict[str, AssertionResult] | None = None,
        suite_name: str = "default",
    ) -> str:
        """Format results as human-readable plaintext for terminal output."""
        lines: list[str] = []
        lines.append(f"LLMAssert Report: {suite_name}")
        lines.append("=" * 60)
        lines.append("")

        if suite_result:
            status = "PASSED" if suite_result.passed else "FAILED"
            lines.append(f"Suite: {status}")
            lines.append(
                f"  {suite_result.passed_cases}/{suite_result.total_cases} cases passed "
                f"({suite_result.execution_time_ms}ms)"
            )
            lines.append("")

            for case_name, case_result in suite_result.case_results.items():
                lines.extend(_format_case_plaintext(case_name, case_result))

        if assertion_results:
            for name, result in assertion_results.items():
                lines.extend(_format_case_plaintext(name, result))

        return "\n".join(lines)

    @staticmethod
    def to_junit(
        suite_result: SuiteResult | None = None,
        assertion_results: dict[str, AssertionResult] | None = None,
        suite_name: str = "default",
    ) -> str:
        """Format results as JUnit XML for CI integration."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']

        cases: dict[str, AssertionResult] = {}
        if suite_result:
            cases = suite_result.case_results
        elif assertion_results:
            cases = assertion_results

        total = len(cases)
        failures = sum(1 for r in cases.values() if not r.passed)
        total_time_ms = sum(r.execution_time_ms for r in cases.values())

        lines.append(
            f'<testsuites name="{_xml_escape(suite_name)}" '
            f'tests="{total}" failures="{failures}" '
            f'time="{total_time_ms / 1000:.3f}">'
        )
        lines.append(
            f'  <testsuite name="{_xml_escape(suite_name)}" '
            f'tests="{total}" failures="{failures}">'
        )

        for case_name, case_result in cases.items():
            lines.append(
                f'    <testcase name="{_xml_escape(case_name)}" '
                f'time="{case_result.execution_time_ms / 1000:.3f}">'
            )

            if not case_result.passed:
                failure_messages = [
                    a.message for a in case_result.assertions if not a.passed
                ]
                lines.append(
                    f'      <failure message="{_xml_escape("; ".join(failure_messages))}" />'
                )

            lines.append("    </testcase>")

        lines.append("  </testsuite>")
        lines.append("</testsuites>")
        return "\n".join(lines)


def _serialize_assertion_result(result: AssertionResult) -> dict[str, Any]:
    """Convert an AssertionResult to a JSON-serializable dict."""
    serialized: dict[str, Any] = {
        "passed": result.passed,
        "execution_time_ms": result.execution_time_ms,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "assertions": [
            {
                "assertion_type": a.assertion_type,
                "assertion_name": a.assertion_name,
                "passed": a.passed,
                "score": a.score,
                "threshold": a.threshold,
                "confidence": a.confidence,
                "message": a.message,
            }
            for a in result.assertions
        ],
    }

    # Include trajectory data when tool_calls are present on the provider response
    if result.provider_response and result.provider_response.tool_calls:
        serialized["tool_calls"] = result.provider_response.tool_calls

    return serialized


def _format_case_plaintext(case_name: str, result: AssertionResult) -> list[str]:
    """Format a single case result as plaintext lines."""
    lines: list[str] = []
    status = "PASS" if result.passed else "FAIL"
    lines.append(f"  [{status}] {case_name} ({result.execution_time_ms}ms)")
    lines.append(f"         model: {result.model}")

    for individual in result.assertions:
        marker = "+" if individual.passed else "X"
        score_str = ""
        if individual.score is not None:
            score_str = f" (score={individual.score:.4f}"
            if individual.threshold is not None:
                score_str += f", threshold={individual.threshold:.4f}"
            score_str += ")"

        lines.append(
            f"    [{marker}] {individual.assertion_name}{score_str}"
        )

        if not individual.passed:
            lines.append(f"        {individual.message}")

    lines.append("")
    return lines


def render_rich_report(
    suite_result: SuiteResult | None = None,
    assertion_results: dict[str, AssertionResult] | None = None,
    suite_name: str = "default",
) -> None:
    """Render results directly to the Rich console with styled output.

    This is the primary display path for interactive terminal sessions.
    Falls through to plaintext when the console is not a TTY.
    """
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    from llm_assert.cli.console import (
        console,
    )

    if suite_result:
        # Suite header panel
        if suite_result.passed:
            status_text = Text("PASSED", style="llm_assert.pass")
        else:
            status_text = Text("FAILED", style="llm_assert.fail")

        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="llm_assert.key")
        summary.add_column()
        summary.add_row("Status", status_text)
        summary.add_row(
            "Cases",
            f"{suite_result.passed_cases}/{suite_result.total_cases} passed",
        )
        summary.add_row("Duration", f"{suite_result.execution_time_ms}ms")

        console.print(Panel(
            summary,
            title=f"[llm_assert.header]{suite_name}[/llm_assert.header]",
            border_style="llm_assert.header",
            padding=(0, 1),
        ))
        console.print()

        for case_name, case_result in suite_result.case_results.items():
            _render_rich_case(case_name, case_result)

    if assertion_results:
        for name, result in assertion_results.items():
            _render_rich_case(name, result)


def _render_rich_case(case_name: str, result: AssertionResult) -> None:
    """Render a single case with a tree of assertion results."""
    from rich.markup import escape
    from rich.tree import Tree

    from llm_assert.cli.console import (
        FAIL_MARKER,
        PASS_MARKER,
        console,
        format_score,
    )

    marker = PASS_MARKER if result.passed else FAIL_MARKER
    timing = f"{result.execution_time_ms}ms, {escape(result.model)}"
    tree = Tree(
        f"{marker} [llm_assert.key]{escape(case_name)}[/llm_assert.key]"
        f"  [llm_assert.muted]({timing})[/llm_assert.muted]"
    )

    for individual in result.assertions:
        a_marker = PASS_MARKER if individual.passed else FAIL_MARKER

        label_parts = [f"{a_marker} {escape(individual.assertion_name)}"]

        if individual.score is not None:
            label_parts.append(
                f"  {format_score(individual.score, individual.threshold)}"
            )
            if individual.threshold is not None:
                label_parts.append(
                    f" [llm_assert.muted](threshold {individual.threshold:.4f})[/llm_assert.muted]"
                )

        node = tree.add("".join(label_parts))

        if not individual.passed and individual.message:
            node.add(f"[llm_assert.muted]{escape(individual.message)}[/llm_assert.muted]")

    console.print(tree)


def _xml_escape(text: str) -> str:
    """Escape special characters for XML attribute values."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
