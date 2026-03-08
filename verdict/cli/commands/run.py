"""verdict run: execute a YAML-defined assertion suite."""

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
def run(suite_file: str, provider: str | None, output_format: str, output: str | None, strict: bool) -> None:
    """Execute a YAML assertion suite and produce a structured report.

    Exits with non-zero code on any assertion failure, making this
    the interface CI pipelines use.
    """
    from rich.markup import escape

    from verdict.cli.console import console
    from verdict.core.report import ReportFormatter, render_rich_report
    from verdict.core.yaml_suite import load_yaml_suite
    from verdict.errors import SuiteParseError, VerdictError

    try:
        suite = load_yaml_suite(suite_file)
    except SuiteParseError as parse_error:
        console.print(f"[verdict.fail]Error:[/verdict.fail] {escape(str(parse_error))}")
        sys.exit(2)

    if strict:
        suite.config.strict_mode = True

    # Resolve provider
    resolved_provider = _resolve_provider(provider, suite)
    if resolved_provider is None:
        console.print(
            "[verdict.fail]Error:[/verdict.fail] No provider specified. Use --provider flag or set "
            "'provider' in the suite file.",
            highlight=False,
        )
        sys.exit(2)

    from verdict.core.runner import AssertionRunner

    runner = AssertionRunner(provider=resolved_provider, config=suite.config)

    try:
        with console.status("[verdict.muted]Running suite...[/verdict.muted]", spinner="dots"):
            suite_result = runner.run_suite(suite)
    except VerdictError as verdict_error:
        console.print(f"[verdict.fail]Error during suite execution:[/verdict.fail] {escape(str(verdict_error))}")
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
            report_text = ReportFormatter.to_plaintext(suite_result=suite_result, suite_name=suite.name)
        with open(output, "w") as fh:
            fh.write(report_text)
        console.print(f"Report written to [verdict.key]{output}[/verdict.key]")
    elif report_text is not None:
        # JSON or JUnit: emit raw text (no Rich markup)
        click.echo(report_text)
    else:
        # Plaintext to terminal: use Rich rendering
        render_rich_report(suite_result=suite_result, suite_name=suite.name)

    if not suite_result.passed:
        sys.exit(1)


def _resolve_provider(provider_name: str | None, suite):
    """Resolve a provider from the CLI flag, suite config, or environment."""
    import os

    from rich.markup import escape

    from verdict.cli.console import console

    name = provider_name or os.environ.get("VERDICT_PROVIDER")

    if not name:
        return None

    name = name.lower().strip()

    from verdict.providers.mock import MockProvider

    if name == "mock":
        return MockProvider(response_fn=lambda prompt, msgs=None: prompt)

    provider_map = {
        "openai": ("verdict.providers.openai", "OpenAIProvider"),
        "anthropic": ("verdict.providers.anthropic", "AnthropicProvider"),
        "google": ("verdict.providers.google", "GoogleProvider"),
        "mistral": ("verdict.providers.mistral", "MistralProvider"),
        "ollama": ("verdict.providers.ollama", "OllamaProvider"),
        "litellm": ("verdict.providers.litellm", "LiteLLMProvider"),
    }

    if name not in provider_map:
        console.print(
            f"[verdict.fail]Unknown provider '{name}'.[/verdict.fail] "
            f"Available: {', '.join(sorted(provider_map.keys()))}, mock",
        )
        return None

    module_path, class_name = provider_map[name]
    try:
        import importlib
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
        return provider_class()
    except ImportError:
        console.print(
            f"[verdict.fail]Provider '{name}' requires additional dependencies.[/verdict.fail] "
            f"Install with: pip install verdict[{name}]",
        )
        return None
    except Exception as init_error:
        console.print(
            f"[verdict.fail]Failed to initialize provider '{escape(name)}':[/verdict.fail] {escape(str(init_error))}",
        )
        return None
