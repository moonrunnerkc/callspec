"""callspec providers: list installed provider extras and check connectivity."""

from __future__ import annotations

import importlib
import time

import click

PROVIDER_MAP = {
    "openai": ("callspec.providers.openai", "OpenAIProvider"),
    "anthropic": ("callspec.providers.anthropic", "AnthropicProvider"),
    "google": ("callspec.providers.google", "GoogleProvider"),
    "mistral": ("callspec.providers.mistral", "MistralProvider"),
    "ollama": ("callspec.providers.ollama", "OllamaProvider"),
    "litellm": ("callspec.providers.litellm", "LiteLLMProvider"),
}

# Maps provider names to the SDK package that must be importable
# for the provider to actually function.
PROVIDER_SDK_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "google.generativeai",
    "mistral": "mistralai",
    "ollama": "ollama",
    "litellm": "litellm",
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

    from callspec.cli.console import console

    table = Table(title="Callspec Providers", border_style="dim", title_style="callspec.header")
    table.add_column("Provider", style="callspec.key")
    table.add_column("Status")
    if run_check:
        table.add_column("Model", style="callspec.muted")
        table.add_column("Latency", style="callspec.muted", justify="right")

    any_installed = False

    for name in sorted(PROVIDER_MAP.keys()):
        module_path, class_name = PROVIDER_MAP[name]

        # Check whether the underlying SDK package is importable, not just
        # the callspec adapter module (which always exists in the package).
        # Provider __init__ is lazy: SDK import happens on first call, not
        # on construction. We check the SDK package directly.
        sdk_module = PROVIDER_SDK_MAP.get(name, "")
        try:
            importlib.import_module(sdk_module)
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
            installed = True
        except (ImportError, Exception):
            installed = False

        if not installed:
            row = [name, "[callspec.skip]NOT INSTALLED[/callspec.skip]"]
            if run_check:
                row += ["", ""]
            table.add_row(*row)
            continue

        any_installed = True

        if not run_check:
            table.add_row(name, "[callspec.pass]INSTALLED[/callspec.pass]")
            continue

        # Connectivity check
        try:
            provider_instance = provider_class()
        except Exception as init_err:
            from rich.markup import escape
            table.add_row(
                name,
                "[callspec.fail]INIT FAILED[/callspec.fail]",
                escape(str(init_err)[:40]),
                "",
            )
            continue

        start = time.monotonic()
        try:
            with console.status(
                f"[callspec.muted]Checking {name}...[/callspec.muted]",
                spinner="dots",
            ):
                response = provider_instance.call(MINIMAL_PROMPT)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            model_id = getattr(response, "model", "unknown")
            table.add_row(
                name,
                "[callspec.pass]OK[/callspec.pass]",
                model_id,
                f"{elapsed_ms}ms",
            )
        except Exception as call_err:
            from rich.markup import escape
            elapsed_ms = int((time.monotonic() - start) * 1000)
            table.add_row(
                name,
                "[callspec.fail]CALL FAILED[/callspec.fail]",
                escape(str(call_err)[:40]),
                f"{elapsed_ms}ms",
            )

    # Mock provider is always available
    row = ["mock", "[callspec.pass]BUILT-IN[/callspec.pass]"]
    if run_check:
        row += ["", ""]
    table.add_row(*row)

    console.print(table)

    if not any_installed:
        console.print(
            "\n[callspec.warn]No external providers installed."
            "[/callspec.warn] Install at least one:\n"
            "  pip install callspec\\[openai]\n"
            "  pip install callspec\\[anthropic]\n"
            "  pip install callspec\\[ollama]"
        )
