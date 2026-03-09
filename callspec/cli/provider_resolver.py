"""Shared provider resolution logic for CLI commands.

Resolves a provider instance from a name string, falling back to the
CALLSPEC_PROVIDER environment variable. Used by both `callspec run`
and `callspec snapshot` to avoid duplicating the resolution map.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from callspec.providers.base import BaseProvider

logger = logging.getLogger(__name__)

_PROVIDER_MAP: dict[str, tuple[str, str]] = {
    "openai": ("callspec.providers.openai", "OpenAIProvider"),
    "anthropic": ("callspec.providers.anthropic", "AnthropicProvider"),
    "google": ("callspec.providers.google", "GoogleProvider"),
    "mistral": ("callspec.providers.mistral", "MistralProvider"),
    "ollama": ("callspec.providers.ollama", "OllamaProvider"),
    "litellm": ("callspec.providers.litellm", "LiteLLMProvider"),
}


def resolve_provider(
    provider_name: str | None,
    *,
    require: bool = False,
) -> BaseProvider | None:
    """Resolve a provider by name, env var, or return None.

    Args:
        provider_name: Explicit provider name from a CLI flag.
        require: If True, print an error and return None when no provider
            can be resolved. If False, return None silently.

    Returns:
        A configured BaseProvider instance, or None on failure.
    """
    from rich.markup import escape

    from callspec.cli.console import console

    name = provider_name or os.environ.get("CALLSPEC_PROVIDER")

    if not name:
        if require:
            console.print(
                "[callspec.fail]No provider specified.[/callspec.fail] "
                "Use --provider flag or set CALLSPEC_PROVIDER env var.",
            )
        return None

    name = name.lower().strip()

    if name == "mock":
        from callspec.providers.mock import MockProvider

        return MockProvider(response_fn=lambda prompt, msgs=None: prompt)

    if name not in _PROVIDER_MAP:
        console.print(
            f"[callspec.fail]Unknown provider '{escape(name)}'.[/callspec.fail] "
            f"Available: {', '.join(sorted(_PROVIDER_MAP.keys()))}, mock",
        )
        return None

    module_path, class_name = _PROVIDER_MAP[name]
    try:
        module = importlib.import_module(module_path)
        provider_class: type[Any] = getattr(module, class_name)
        return provider_class()  # type: ignore[no-any-return]
    except ImportError:
        console.print(
            f"[callspec.fail]Provider '{escape(name)}' requires "
            f"additional dependencies.[/callspec.fail] "
            f"Install with: pip install callspec[{name}]",
        )
        return None
    except Exception as init_error:
        console.print(
            f"[callspec.fail]Failed to initialize provider "
            f"'{escape(name)}':[/callspec.fail] "
            f"{escape(str(init_error))}",
        )
        return None
