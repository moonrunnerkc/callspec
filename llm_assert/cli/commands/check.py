"""llm-assert check: sanity-check provider connectivity.

The first command new users run after pip install. It must work flawlessly
or the developer experience fails at the first step.
"""

from __future__ import annotations

import sys
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
    from llm_assert.cli.console import console

    if provider:
        targets = [provider.lower().strip()]
        unknown = [t for t in targets if t not in PROVIDER_MAP]
        if unknown:
            console.print(
                f"[llm_assert.fail]Unknown provider:[/llm_assert.fail] {unknown[0]}. "
                f"Available: {', '.join(sorted(PROVIDER_MAP.keys()))}",
            )
            sys.exit(2)
    else:
        targets = _detect_installed_providers()
        if not targets:
            console.print(
                "[llm_assert.fail]No provider extras installed.[/llm_assert.fail] Install one with:\n"
                "  pip install llm-assert[openai]\n"
                "  pip install llm-assert[anthropic]\n"
                "  pip install llm-assert[ollama]\n"
                "  pip install llm-assert[google]\n"
                "  pip install llm-assert[mistral]\n"
                "  pip install llm-assert[litellm]",
            )
            sys.exit(1)

    console.print(f"Checking {len(targets)} provider(s)...\n")

    failures = []
    for name in targets:
        _check_single_provider(name, failures)

    console.print()
    if failures:
        console.print(
            f"[llm_assert.fail]{len(failures)} provider(s) failed connectivity check.[/llm_assert.fail]"
        )
        sys.exit(1)
    else:
        console.print(
            f"[llm_assert.pass]All {len(targets)} provider(s) OK.[/llm_assert.pass]"
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

    from llm_assert.cli.console import FAIL_MARKER, PASS_MARKER, SKIP_MARKER, console

    module_path, class_name = PROVIDER_MAP[name]

    # Phase 1: import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        console.print(
            f"  {SKIP_MARKER} [llm_assert.key]{name}[/llm_assert.key]  "
            f"[llm_assert.skip]not installed "
            f"(pip install llm-assert[{name}])[/llm_assert.skip]"
        )
        return

    # Phase 2: instantiate the provider
    provider_class = getattr(module, class_name)
    try:
        provider_instance = provider_class()
    except Exception as init_err:
        console.print(
            f"  {FAIL_MARKER} [llm_assert.key]{name}[/llm_assert.key]  "
            f"[llm_assert.fail]initialization error:[/llm_assert.fail] "
            f"{escape(str(init_err))}"
        )
        failures.append(name)
        return

    # Phase 3: make a minimal call with a spinner
    start_ms = time.monotonic()
    try:
        with console.status(f"  [llm_assert.muted]Calling {name}...[/llm_assert.muted]", spinner="dots"):
            response = provider_instance.call(MINIMAL_PROMPT)
        elapsed_ms = int((time.monotonic() - start_ms) * 1000)
    except Exception as call_err:
        elapsed_ms = int((time.monotonic() - start_ms) * 1000)
        console.print(
            f"  {FAIL_MARKER} [llm_assert.key]{name}[/llm_assert.key]  "
            f"[llm_assert.fail]call failed[/llm_assert.fail] "
            f"after {elapsed_ms}ms: {escape(str(call_err))}"
        )
        failures.append(name)
        return

    model_id = getattr(response, "model", "unknown")
    console.print(
        f"  {PASS_MARKER} [llm_assert.key]{name}[/llm_assert.key]  "
        f"model={model_id}  [llm_assert.muted]{elapsed_ms}ms[/llm_assert.muted]"
    )
