"""Pytest report hook: adds Callspec-specific metadata to test reports.

When a test uses Callspec assertions, the report hook attaches the
assertion type, score, threshold, provider, and model version to
the test report. This metadata appears in the JSON and JUnit reports
produced by --callspec-report.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from callspec.core.types import AssertionResult

# Stored per-test Callspec results, keyed by node ID
_callspec_results: dict[str, list[AssertionResult]] = {}


def record_callspec_result(node_id: str, result: AssertionResult) -> None:
    """Record a Callspec result for inclusion in the test report.

    Called automatically by the assert_callspec_pass helper, or
    manually by developers who want result tracking without the
    helper function.
    """
    if node_id not in _callspec_results:
        _callspec_results[node_id] = []
    _callspec_results[node_id].append(result)


def get_callspec_results(node_id: str) -> list[AssertionResult]:
    """Retrieve recorded Callspec results for a test node."""
    return _callspec_results.get(node_id, [])


def clear_callspec_results() -> None:
    """Clear all recorded Callspec results. Called at session end."""
    _callspec_results.clear()


def _serialize_assertion_result(result: AssertionResult) -> dict[str, Any]:
    """Convert an AssertionResult to a JSON-serializable dict."""
    serialized: dict[str, Any] = {
        "passed": result.passed,
        "execution_time_ms": result.execution_time_ms,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "assertions": [],
    }

    for individual in result.assertions:
        serialized["assertions"].append({
            "assertion_type": individual.assertion_type,
            "assertion_name": individual.assertion_name,
            "passed": individual.passed,
            "score": individual.score,
            "threshold": individual.threshold,
            "confidence": individual.confidence,
            "message": individual.message,
        })

    return serialized


class CallspecReportPlugin:
    """Pytest plugin that collects Callspec results and writes reports.

    Activated by --callspec-report <format>. Supported formats:
    json, junit. The JSON format is designed for ingestion by
    verdict.run for historical tracking.
    """

    def __init__(self, report_format: str, report_path: str | None = None) -> None:
        self._format = report_format
        self._report_path = report_path

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        """Write the Callspec report after all tests complete."""
        if not _callspec_results:
            return

        if self._format == "json":
            self._write_json_report(session)
        elif self._format == "junit":
            self._write_junit_report(session)

    def _write_json_report(self, session: pytest.Session) -> None:
        """Produce a JSON report with all Callspec assertion results."""
        report: dict[str, Any] = {
            "callspec_version": "0.1.0",
            "total_tests": len(_callspec_results),
            "tests": {},
        }

        for node_id, results in _callspec_results.items():
            report["tests"][node_id] = [
                _serialize_assertion_result(r) for r in results
            ]

        output_path = self._report_path or "callspec_report.json"
        Path(output_path).write_text(json.dumps(report, indent=2))

    def _write_junit_report(self, session: pytest.Session) -> None:
        """Produce a JUnit XML report with Callspec metadata in properties."""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append('<testsuites name="callspec">')

        total_tests = 0
        total_failures = 0

        for node_id, results in _callspec_results.items():
            for result in results:
                total_tests += 1
                if not result.passed:
                    total_failures += 1

        lines.append(
            f'  <testsuite name="callspec" tests="{total_tests}" '
            f'failures="{total_failures}">'
        )

        for node_id, results in _callspec_results.items():
            for result in results:
                lines.append(
                    f'    <testcase name="{_xml_escape(node_id)}" '
                    f'time="{result.execution_time_ms / 1000:.3f}">'
                )

                # Callspec metadata as properties
                lines.append("      <properties>")
                lines.append(
                    f'        <property name="model" '
                    f'value="{_xml_escape(result.model)}" />'
                )
                for individual in result.assertions:
                    lines.append(
                        f'        <property name="{individual.assertion_name}" '
                        f'value="{"pass" if individual.passed else "fail"}" />'
                    )
                    if individual.score is not None:
                        lines.append(
                            f'        <property name="{individual.assertion_name}.score" '
                            f'value="{individual.score:.4f}" />'
                        )
                lines.append("      </properties>")

                if not result.passed:
                    failure_messages = [
                        i.message for i in result.assertions if not i.passed
                    ]
                    lines.append(
                        f'      <failure message="{_xml_escape("; ".join(failure_messages))}" />'
                    )

                lines.append("    </testcase>")

        lines.append("  </testsuite>")
        lines.append("</testsuites>")

        output_path = self._report_path or "callspec_report.xml"
        Path(output_path).write_text("\n".join(lines))


def _xml_escape(text: str) -> str:
    """Escape special characters for XML attribute values."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )
