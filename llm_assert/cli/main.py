"""LLMAssert CLI entry point.

click-based CLI that provides:
    llm-assert run <suite>     - execute a YAML assertion suite
    llm-assert check           - sanity check provider connectivity
    llm-assert snapshot        - manage baseline snapshots
    llm-assert report          - pretty-print a saved result JSON
    llm-assert providers       - list installed providers, check connectivity
"""

from __future__ import annotations

import click

from llm_assert.version import __version__


@click.group()
@click.version_option(version=__version__, prog_name="llm-assert")
def cli() -> None:
    """LLMAssert: behavioral assertion testing for LLM applications."""
    pass


# Import and register command groups after cli is defined
# to avoid circular imports
from llm_assert.cli.commands.check import check  # noqa: E402
from llm_assert.cli.commands.providers import providers  # noqa: E402
from llm_assert.cli.commands.report import report  # noqa: E402
from llm_assert.cli.commands.run import run  # noqa: E402
from llm_assert.cli.commands.snapshot import snapshot  # noqa: E402

cli.add_command(run)
cli.add_command(check)
cli.add_command(snapshot)
cli.add_command(report)
cli.add_command(providers)
