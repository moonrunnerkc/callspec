"""Callspec CLI entry point.

click-based CLI that provides:
    callspec run <suite>     - execute a YAML assertion suite
    callspec check           - sanity check provider connectivity
    callspec snapshot        - manage baseline snapshots
    callspec report          - pretty-print a saved result JSON
    callspec providers       - list installed providers, check connectivity
"""

from __future__ import annotations

import click

from callspec.version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="callspec")
def cli() -> None:
    """Callspec: behavioral assertion testing for LLM applications."""
    pass


# Import and register command groups after cli is defined
# to avoid circular imports
from callspec.cli.commands.check import check  # noqa: E402
from callspec.cli.commands.providers import providers  # noqa: E402
from callspec.cli.commands.report import report  # noqa: E402
from callspec.cli.commands.run import run  # noqa: E402
from callspec.cli.commands.snapshot import snapshot  # noqa: E402

cli.add_command(run)
cli.add_command(check)
cli.add_command(snapshot)
cli.add_command(report)
cli.add_command(providers)
