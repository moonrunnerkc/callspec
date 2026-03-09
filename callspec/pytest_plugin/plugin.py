"""Callspec pytest plugin entry point.

Registers fixtures, custom marks, CLI flags, and report hooks with pytest.
Discovered via the pytest11 entry point group defined in pyproject.toml.

This module is deliberately thin: it imports and delegates to the fixture,
assertion, and reporter modules. The separation keeps each concern testable
in isolation.
"""

from __future__ import annotations

import pytest

# Re-export fixtures so pytest discovers them via this entry point module
from callspec.pytest_plugin.fixtures import (  # noqa: F401
    callspec_config,
    callspec_provider,
    callspec_runner,
    trajectory_runner,
)
from callspec.pytest_plugin.reporter import CallspecReportPlugin, clear_callspec_results


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register Callspec-specific CLI flags with pytest."""
    group = parser.getgroup("callspec", "Callspec behavioral assertion testing")

    group.addoption(
        "--callspec-snapshot",
        action="store_true",
        default=False,
        help="Run snapshot creation/update operations instead of assertions.",
    )

    group.addoption(
        "--callspec-strict",
        action="store_true",
        default=False,
        help="Treat semantic assertion warnings (score within 5%% of threshold) as failures.",
    )

    group.addoption(
        "--callspec-report",
        action="store",
        default=None,
        metavar="FORMAT",
        help="Produce a Callspec report. Formats: json, junit.",
    )

    group.addoption(
        "--callspec-report-path",
        action="store",
        default=None,
        metavar="PATH",
        help="Output path for the Callspec report file.",
    )

    group.addoption(
        "--callspec-skip-behavioral",
        action="store_true",
        default=False,
        help="Skip tests marked with @pytest.mark.callspec_behavioral.",
    )

    group.addoption(
        "--callspec-skip-contracts",
        action="store_true",
        default=False,
        help="Skip tests marked with @pytest.mark.tool_contract.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register markers and activate report plugin if requested."""
    config.addinivalue_line(
        "markers",
        "callspec_behavioral: marks tests as behavioral (multi-sample, expensive)",
    )
    config.addinivalue_line(
        "markers",
        "tool_contract: marks tests as tool-call contract tests",
    )

    # Activate report plugin when --callspec-report is specified
    report_format = config.getoption("--callspec-report", default=None)
    if report_format:
        report_path = config.getoption("--callspec-report-path", default=None)
        config.pluginmanager.register(
            CallspecReportPlugin(report_format, report_path),
            name="callspec-report",
        )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip behavioral or contract tests when corresponding flags are active."""
    skip_behavioral = config.getoption("--callspec-skip-behavioral", default=False)
    skip_contracts = config.getoption("--callspec-skip-contracts", default=False)

    if not skip_behavioral and not skip_contracts:
        return

    behavioral_skip = pytest.mark.skip(
        reason="Skipped by --callspec-skip-behavioral (expensive multi-sample test)"
    )
    contract_skip = pytest.mark.skip(
        reason="Skipped by --callspec-skip-contracts (tool-call contract test)"
    )

    for test_item in items:
        if skip_behavioral and "callspec_behavioral" in test_item.keywords:
            test_item.add_marker(behavioral_skip)
        if skip_contracts and "tool_contract" in test_item.keywords:
            test_item.add_marker(contract_skip)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Clean up recorded Callspec results at session end."""
    clear_callspec_results()
