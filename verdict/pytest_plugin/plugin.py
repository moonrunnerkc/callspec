"""Verdict pytest plugin entry point.

Registers fixtures, custom marks, CLI flags, and report hooks with pytest.
Discovered via the pytest11 entry point group defined in pyproject.toml.

This module is deliberately thin: it imports and delegates to the fixture,
assertion, and reporter modules. The separation keeps each concern testable
in isolation.
"""

from __future__ import annotations

import pytest

# Re-export fixtures so pytest discovers them via this entry point module
from verdict.pytest_plugin.fixtures import (  # noqa: F401
    verdict_config,
    verdict_provider,
    verdict_runner,
)
from verdict.pytest_plugin.reporter import VerdictReportPlugin, clear_verdict_results


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register Verdict-specific CLI flags with pytest."""
    group = parser.getgroup("verdict", "Verdict behavioral assertion testing")

    group.addoption(
        "--verdict-snapshot",
        action="store_true",
        default=False,
        help="Run snapshot creation/update operations instead of assertions.",
    )

    group.addoption(
        "--verdict-strict",
        action="store_true",
        default=False,
        help="Treat semantic assertion warnings (score within 5%% of threshold) as failures.",
    )

    group.addoption(
        "--verdict-report",
        action="store",
        default=None,
        metavar="FORMAT",
        help="Produce a Verdict report. Formats: json, junit.",
    )

    group.addoption(
        "--verdict-report-path",
        action="store",
        default=None,
        metavar="PATH",
        help="Output path for the Verdict report file.",
    )

    group.addoption(
        "--verdict-skip-behavioral",
        action="store_true",
        default=False,
        help="Skip tests marked with @pytest.mark.verdict_behavioral.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register markers and activate report plugin if requested."""
    config.addinivalue_line(
        "markers",
        "verdict_behavioral: marks tests as behavioral (multi-sample, expensive)",
    )

    # Activate report plugin when --verdict-report is specified
    report_format = config.getoption("--verdict-report", default=None)
    if report_format:
        report_path = config.getoption("--verdict-report-path", default=None)
        config.pluginmanager.register(
            VerdictReportPlugin(report_format, report_path),
            name="verdict-report",
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip behavioral tests when --verdict-skip-behavioral is active."""
    if not config.getoption("--verdict-skip-behavioral", default=False):
        return

    skip_marker = pytest.mark.skip(
        reason="Skipped by --verdict-skip-behavioral (expensive multi-sample test)"
    )
    for test_item in items:
        if "verdict_behavioral" in test_item.keywords:
            test_item.add_marker(skip_marker)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Clean up recorded Verdict results at session end."""
    clear_verdict_results()
