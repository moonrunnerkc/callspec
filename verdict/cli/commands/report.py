"""verdict report: pretty-print a saved Verdict result JSON."""

from __future__ import annotations

import json
import sys

import click


@click.command()
@click.argument("report_file", type=click.Path(exists=True))
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["plaintext", "json", "junit"]),
    default="plaintext",
    help="Output format. Default is plaintext for terminal display.",
)
def report(report_file: str, output_format: str) -> None:
    """Pretty-print a saved Verdict result JSON file.

    Reads a JSON report produced by 'verdict run --format json' or
    'pytest --verdict-report json' and renders it for human consumption.
    """
    from rich.markup import escape

    from verdict.cli.console import console

    try:
        with open(report_file) as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as json_err:
        console.print(
            f"[verdict.fail]Invalid JSON[/verdict.fail] "
            f"in {escape(report_file)}: {escape(str(json_err))}"
        )
        sys.exit(2)
    except OSError as io_err:
        console.print(
            f"[verdict.fail]Cannot read[/verdict.fail] "
            f"{escape(report_file)}: {escape(str(io_err))}"
        )
        sys.exit(2)

    if output_format == "json":
        # Re-emit with consistent formatting, no Rich markup
        click.echo(json.dumps(raw, indent=2, default=str))
        return

    if output_format == "junit":
        _render_junit_from_raw(raw)
        return

    # Rich plaintext rendering
    _render_plaintext(raw)


def _render_plaintext(raw: dict) -> None:
    """Render a saved JSON report with Rich formatting."""
    from rich.markup import escape
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    from verdict.cli.console import (
        FAIL_MARKER,
        PASS_MARKER,
        console,
        format_score,
    )

    suite_name = raw.get("suite_name", raw.get("suite", "Unknown Suite"))
    passed = raw.get("passed", None)
    total_ms = raw.get("execution_time_ms", 0)

    if passed:
        status_text = "[verdict.pass]PASSED[/verdict.pass]"
    elif passed is not None:
        status_text = "[verdict.fail]FAILED[/verdict.fail]"
    else:
        status_text = "[verdict.warn]UNKNOWN[/verdict.warn]"

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="verdict.key")
    summary.add_column()
    summary.add_row("Status", status_text)
    summary.add_row("Duration", f"{total_ms}ms")

    console.print(Panel(
        summary,
        title=f"[verdict.header]{escape(suite_name)}[/verdict.header]",
        border_style="verdict.header",
        padding=(0, 1),
    ))
    console.print()

    cases = raw.get("cases", raw.get("results", []))
    for case_data in cases:
        case_name = case_data.get("name", case_data.get("case", "unnamed"))
        case_passed = case_data.get("passed", None)
        marker = PASS_MARKER if case_passed else FAIL_MARKER

        tree = Tree(f"{marker} [verdict.key]{escape(case_name)}[/verdict.key]")

        assertions = case_data.get("assertions", [])
        for assertion_data in assertions:
            a_name = assertion_data.get("assertion_name", assertion_data.get("name", "?"))
            a_type = assertion_data.get("assertion_type", assertion_data.get("type", ""))
            a_passed = assertion_data.get("passed", None)
            a_marker = PASS_MARKER if a_passed else FAIL_MARKER

            label_parts = [f"{a_marker} {escape(a_type)}/{escape(a_name)}"]

            score = assertion_data.get("score")
            threshold = assertion_data.get("threshold")
            if score is not None:
                label_parts.append(f"  {format_score(score, threshold)}")
            if threshold is not None:
                label_parts.append(
                    f" [verdict.muted](threshold {threshold})[/verdict.muted]"
                )

            node = tree.add("".join(label_parts))

            message = assertion_data.get("message", "")
            if message and not a_passed:
                node.add(f"[verdict.muted]{escape(message)}[/verdict.muted]")

        console.print(tree)

    console.print()
    total_cases = len(cases)
    passed_cases = sum(1 for c in cases if c.get("passed"))
    failed_cases = total_cases - passed_cases
    console.print(
        f"[verdict.key]Total:[/verdict.key] {total_cases} cases, "
        f"[verdict.pass]{passed_cases} passed[/verdict.pass], "
        f"[verdict.fail]{failed_cases} failed[/verdict.fail]"
    )


def _render_junit_from_raw(raw: dict) -> None:
    """Convert a saved JSON report to JUnit XML and print it."""
    suite_name = _xml_escape(raw.get("suite_name", "verdict"))
    cases = raw.get("cases", raw.get("results", []))
    total = len(cases)
    failures = sum(1 for c in cases if not c.get("passed"))
    total_ms = raw.get("execution_time_ms", 0)
    total_seconds = total_ms / 1000.0

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuite name="{suite_name}" tests="{total}" '
        f'failures="{failures}" time="{total_seconds:.3f}">',
    ]

    for case_data in cases:
        case_name = _xml_escape(case_data.get("name", "unnamed"))
        case_time = case_data.get("execution_time_ms", 0) / 1000.0
        lines.append(f'  <testcase name="{case_name}" time="{case_time:.3f}">')
        if not case_data.get("passed"):
            message_parts = []
            for a in case_data.get("assertions", []):
                if not a.get("passed"):
                    msg = a.get("message", a.get("assertion_name", "failed"))
                    message_parts.append(msg)
            combined = "; ".join(message_parts)
            lines.append(f'    <failure message="{_xml_escape(combined)}"/>')
        lines.append("  </testcase>")

    lines.append("</testsuite>")
    click.echo("\n".join(lines))


def _xml_escape(text: str) -> str:
    """Escape XML special characters."""
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
