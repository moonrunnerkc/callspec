"""llm-assert providers: list installed provider extras and check connectivity."""

from __future__ import annotations

import importlib
import time

import click

PROVIDER_MAP = {
    "openai": ("llm_assert.providers.openai", "OpenAIProvider"),
    "anthropic": ("llm_assert.providers.anthropic", "AnthropicProvider"),
    "google": ("llm_assert.providers.google", "GoogleProvider"),
    "mistral": ("llm_assert.providers.mistral", "MistralProvider"),
    "ollama": ("llm_assert.providers.ollama", "OllamaProvider"),
    "litellm": ("llm_assert.providers.litellm", "LiteLLMProvider"),
}

MINIMAL_PROMPT = "Respond with exactly: OK"


@click.command()
@click.option(
    "--check", "-c", "run_check",
    is_flag=True,
    default=False,
    help="Also verify connectivity by making a minimal call to each installed provider.",
)
def providers(run_check: bool) -> None:
    """List installed provider extras and their status.

    Shows which provider packages are importable and optionally
    tests connectivity with a minimal call.
    """
    from rich.table import Table

    from llm_assert.cli.console import console

    table = Table(title="LLMAssert Providers", border_style="dim", title_style="llm_assert.header")
    table.add_column("Provider", style="llm_assert.key")
    table.add_column("Status")
    if run_check:
        table.add_column("Model", style="llm_assert.muted")
        table.add_column("Latency", style="llm_assert.muted", justify="right")

    any_installed = False

    for name in sorted(PROVIDER_MAP.keys()):
        module_path, class_name = PROVIDER_MAP[name]

        try:
            module = importlib.import_module(module_path)
            installed = True
        except ImportError:
            installed = False

        if not installed:
            row = [name, "[llm_assert.skip]NOT INSTALLED[/llm_assert.skip]"]
            if run_check:
                row += ["", ""]
            table.add_row(*row)
            continue

        any_installed = True

        if not run_check:
            table.add_row(name, "[llm_assert.pass]INSTALLED[/llm_assert.pass]")
            continue

        # Connectivity check
        provider_class = getattr(module, class_name)
        try:
            provider_instance = provider_class()
        except Exception as init_err:
            from rich.markup import escape
            table.add_row(
                name,
                "[llm_assert.fail]INIT FAILED[/llm_assert.fail]",
                escape(str(init_err)[:40]),
                "",
            )
            continue

        start = time.monotonic()
        try:
            with console.status(
                f"[llm_assert.muted]Checking {name}...[/llm_assert.muted]",
                spinner="dots",
            ):
                response = provider_instance.call(MINIMAL_PROMPT)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            model_id = getattr(response, "model", "unknown")
            table.add_row(
                name,
                "[llm_assert.pass]OK[/llm_assert.pass]",
                model_id,
                f"{elapsed_ms}ms",
            )
        except Exception as call_err:
            from rich.markup import escape
            elapsed_ms = int((time.monotonic() - start) * 1000)
            table.add_row(
                name,
                "[llm_assert.fail]CALL FAILED[/llm_assert.fail]",
                escape(str(call_err)[:40]),
                f"{elapsed_ms}ms",
            )

    # Mock provider is always available
    row = ["mock", "[llm_assert.pass]BUILT-IN[/llm_assert.pass]"]
    if run_check:
        row += ["", ""]
    table.add_row(*row)

    console.print(table)

    if not any_installed:
        console.print(
            "\n[llm_assert.warn]No external providers installed."
            "[/llm_assert.warn] Install at least one:\n"
            "  pip install llm-assert[openai]\n"
            "  pip install llm-assert[anthropic]\n"
            "  pip install llm-assert[ollama]"
        )
