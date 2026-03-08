"""verdict check: sanity-check provider connectivity.

The first command new users run after pip install. It must work flawlessly
or the developer experience fails at the first step.
"""

from __future__ import annotations

import sys
import time

import click

PROVIDER_MAP = {
    "openai": ("verdict.providers.openai", "OpenAIProvider"),
    "anthropic": ("verdict.providers.anthropic", "AnthropicProvider"),
    "google": ("verdict.providers.google", "GoogleProvider"),
    "mistral": ("verdict.providers.mistral", "MistralProvider"),
    "ollama": ("verdict.providers.ollama", "OllamaProvider"),
    "litellm": ("verdict.providers.litellm", "LiteLLMProvider"),
}

MINIMAL_PROMPT = "Respond with exactly: OK"


@click.command()
@click.option(
    "--provider", "-p",
    default=None,
    help="Specific provider to check. Checks all installed providers if not specified.",
)
def check(provider: str | None) -> None:
    """Verify provider connectivity and configuration.

    Invokes each configured provider with a minimal prompt and confirms
    the adapter is reachable and configured correctly. Use this after
    installation to validate your environment.
    """
    from verdict.cli.console import console

    if provider:
        targets = [provider.lower().strip()]
        unknown = [t for t in targets if t not in PROVIDER_MAP]
        if unknown:
            console.print(
                f"[verdict.fail]Unknown provider:[/verdict.fail] {unknown[0]}. "
                f"Available: {', '.join(sorted(PROVIDER_MAP.keys()))}",
            )
            sys.exit(2)
    else:
        targets = _detect_installed_providers()
        if not targets:
            console.print(
                "[verdict.fail]No provider extras installed.[/verdict.fail] Install one with:\n"
                "  pip install verdict[openai]\n"
                "  pip install verdict[anthropic]\n"
                "  pip install verdict[ollama]\n"
                "  pip install verdict[google]\n"
                "  pip install verdict[mistral]\n"
                "  pip install verdict[litellm]",
            )
            sys.exit(1)

    console.print(f"Checking {len(targets)} provider(s)...\n")

    failures = []
    for name in targets:
        _check_single_provider(name, failures)

    console.print()
    if failures:
        console.print(
            f"[verdict.fail]{len(failures)} provider(s) failed connectivity check.[/verdict.fail]"
        )
        sys.exit(1)
    else:
        console.print(
            f"[verdict.pass]All {len(targets)} provider(s) OK.[/verdict.pass]"
        )


def _detect_installed_providers() -> list[str]:
    """Return names of providers whose dependencies are importable."""
    import importlib

    installed = []
    for name, (module_path, _) in PROVIDER_MAP.items():
        try:
            importlib.import_module(module_path)
            installed.append(name)
        except ImportError:
            continue
    return installed


def _check_single_provider(name: str, failures: list[str]) -> None:
    """Attempt to initialize and call a single provider."""
    import importlib

    from rich.markup import escape

    from verdict.cli.console import FAIL_MARKER, PASS_MARKER, SKIP_MARKER, console

    module_path, class_name = PROVIDER_MAP[name]

    # Phase 1: import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        console.print(
            f"  {SKIP_MARKER} [verdict.key]{name}[/verdict.key]  "
            f"[verdict.skip]not installed "
            f"(pip install verdict[{name}])[/verdict.skip]"
        )
        return

    # Phase 2: instantiate the provider
    provider_class = getattr(module, class_name)
    try:
        provider_instance = provider_class()
    except Exception as init_err:
        console.print(
            f"  {FAIL_MARKER} [verdict.key]{name}[/verdict.key]  "
            f"[verdict.fail]initialization error:[/verdict.fail] "
            f"{escape(str(init_err))}"
        )
        failures.append(name)
        return

    # Phase 3: make a minimal call with a spinner
    start_ms = time.monotonic()
    try:
        with console.status(f"  [verdict.muted]Calling {name}...[/verdict.muted]", spinner="dots"):
            response = provider_instance.call(MINIMAL_PROMPT)
        elapsed_ms = int((time.monotonic() - start_ms) * 1000)
    except Exception as call_err:
        elapsed_ms = int((time.monotonic() - start_ms) * 1000)
        console.print(
            f"  {FAIL_MARKER} [verdict.key]{name}[/verdict.key]  "
            f"[verdict.fail]call failed[/verdict.fail] "
            f"after {elapsed_ms}ms: {escape(str(call_err))}"
        )
        failures.append(name)
        return

    model_id = getattr(response, "model", "unknown")
    console.print(
        f"  {PASS_MARKER} [verdict.key]{name}[/verdict.key]  "
        f"model={model_id}  [verdict.muted]{elapsed_ms}ms[/verdict.muted]"
    )
