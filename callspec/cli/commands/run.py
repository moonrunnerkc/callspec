"""callspec run: execute a YAML-defined assertion suite."""

from __future__ import annotations

import sys

import click


@click.command()
@click.argument("suite_file", type=click.Path(exists=True))
@click.option(
    "--provider", "-p",
    default=None,
    help="Provider name override. Reads from suite file if not specified.",
)
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["plaintext", "json", "junit"]),
    default="plaintext",
    help="Output format for the result report.",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Write report to file instead of stdout.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Treat borderline passes (score within 5%% of threshold) as failures.",
)
def run(
    suite_file: str,
    provider: str | None,
    output_format: str,
    output: str | None,
    strict: bool,
) -> None:
    """Execute a YAML assertion suite and produce a structured report.

    Exits with non-zero code on any assertion failure, making this
    the interface CI pipelines use.
    """
    from rich.markup import escape

    from callspec.cli.console import console
    from callspec.core.report import ReportFormatter, render_rich_report
    from callspec.core.yaml_suite import load_yaml_suite
    from callspec.errors import CallspecError, SuiteParseError

    try:
        suite = load_yaml_suite(suite_file)
    except SuiteParseError as parse_error:
        console.print(f"[callspec.fail]Error:[/callspec.fail] {escape(str(parse_error))}")
        sys.exit(2)

    if strict:
        suite.config.strict_mode = True

    # Resolve provider
    from callspec.cli.provider_resolver import resolve_provider

    resolved_provider = resolve_provider(provider, require=True)
    if resolved_provider is None:
        sys.exit(2)

    from callspec.core.runner import AssertionRunner

    runner = AssertionRunner(provider=resolved_provider, config=suite.config)

    try:
        with console.status(
            "[callspec.muted]Running suite...[/callspec.muted]",
            spinner="dots",
        ):
            suite_result = runner.run_suite(suite)
    except CallspecError as callspec_error:
        console.print(
            f"[callspec.fail]Error during suite execution:[/callspec.fail] "
            f"{escape(str(callspec_error))}"
        )
        sys.exit(1)

    # Format output
    if output_format == "json":
        report_text = ReportFormatter.to_json(suite_result=suite_result, suite_name=suite.name)
    elif output_format == "junit":
        report_text = ReportFormatter.to_junit(suite_result=suite_result, suite_name=suite.name)
    else:
        report_text = None

    if output:
        # Writing to file always uses plaintext string form
        if report_text is None:
            report_text = ReportFormatter.to_plaintext(
                suite_result=suite_result,
                suite_name=suite.name,
            )
        with open(output, "w") as fh:
            fh.write(report_text)
        console.print(f"Report written to [callspec.key]{output}[/callspec.key]")
    elif report_text is not None:
        # JSON or JUnit: emit raw text (no Rich markup)
        click.echo(report_text)
    else:
        # Plaintext to terminal: use Rich rendering
        render_rich_report(suite_result=suite_result, suite_name=suite.name)

    if not suite_result.passed:
        sys.exit(1)



