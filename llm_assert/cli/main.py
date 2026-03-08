"""Verdict CLI entry point.

click-based CLI that provides:
    verdict run <suite>     - execute a YAML assertion suite
    verdict check           - sanity check provider connectivity
    verdict snapshot        - manage baseline snapshots
    verdict report          - pretty-print a saved result JSON
    verdict providers       - list installed providers, check connectivity
"""

from __future__ import annotations

import click

from verdict.version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="verdict")
def cli() -> None:
    """Verdict: behavioral assertion testing for LLM applications."""
    pass


# Import and register command groups after cli is defined
# to avoid circular imports
from verdict.cli.commands.check import check  # noqa: E402
from verdict.cli.commands.providers import providers  # noqa: E402
from verdict.cli.commands.report import report  # noqa: E402
from verdict.cli.commands.run import run  # noqa: E402
from verdict.cli.commands.snapshot import snapshot  # noqa: E402

cli.add_command(run)
cli.add_command(check)
cli.add_command(snapshot)
cli.add_command(report)
cli.add_command(providers)
