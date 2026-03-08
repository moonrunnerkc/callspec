"""LLMAssert pytest plugin entry point.

Registers fixtures, custom marks, CLI flags, and report hooks with pytest.
Discovered via the pytest11 entry point group defined in pyproject.toml.

This module is deliberately thin: it imports and delegates to the fixture,
assertion, and reporter modules. The separation keeps each concern testable
in isolation.
"""

from __future__ import annotations

import pytest

# Re-export fixtures so pytest discovers them via this entry point module
from llm_assert.pytest_plugin.fixtures import (  # noqa: F401
    llm_assert_config,
    llm_assert_provider,
    llm_assert_runner,
)
from llm_assert.pytest_plugin.reporter import LLMAssertReportPlugin, clear_llm_assert_results


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register LLMAssert-specific CLI flags with pytest."""
    group = parser.getgroup("llm-assert", "LLMAssert behavioral assertion testing")

    group.addoption(
        "--llm-assert-snapshot",
        action="store_true",
        default=False,
        help="Run snapshot creation/update operations instead of assertions.",
    )

    group.addoption(
        "--llm-assert-strict",
        action="store_true",
        default=False,
        help="Treat semantic assertion warnings (score within 5%% of threshold) as failures.",
    )

    group.addoption(
        "--llm-assert-report",
        action="store",
        default=None,
        metavar="FORMAT",
        help="Produce a LLMAssert report. Formats: json, junit.",
    )

    group.addoption(
        "--llm-assert-report-path",
        action="store",
        default=None,
        metavar="PATH",
        help="Output path for the LLMAssert report file.",
    )

    group.addoption(
        "--llm-assert-skip-behavioral",
        action="store_true",
        default=False,
        help="Skip tests marked with @pytest.mark.llm_assert_behavioral.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register markers and activate report plugin if requested."""
    config.addinivalue_line(
        "markers",
        "llm_assert_behavioral: marks tests as behavioral (multi-sample, expensive)",
    )

    # Activate report plugin when --llm-assert-report is specified
    report_format = config.getoption("--llm-assert-report", default=None)
    if report_format:
        report_path = config.getoption("--llm-assert-report-path", default=None)
        config.pluginmanager.register(
            LLMAssertReportPlugin(report_format, report_path),
            name="llm-assert-report",
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip behavioral tests when --llm-assert-skip-behavioral is active."""
    if not config.getoption("--llm-assert-skip-behavioral", default=False):
        return

    skip_marker = pytest.mark.skip(
        reason="Skipped by --llm-assert-skip-behavioral (expensive multi-sample test)"
    )
    for test_item in items:
        if "llm_assert_behavioral" in test_item.keywords:
            test_item.add_marker(skip_marker)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Clean up recorded LLMAssert results at session end."""
    clear_llm_assert_results()
