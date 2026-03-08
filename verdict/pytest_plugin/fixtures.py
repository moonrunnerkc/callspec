"""Pytest fixtures for Verdict integration.

verdict_config  (session scope) - VerdictConfig loaded from env/conftest
verdict_provider (session scope) - configured provider instance
verdict_runner  (function scope) - fresh Verdict instance per test

These fixtures form the bridge between pytest and the Verdict API.
Developers use them in test files without manual instantiation.
"""

from __future__ import annotations

import os
from typing import Optional

import pytest

from verdict.core.config import VerdictConfig
from verdict.providers.base import BaseProvider
from verdict.providers.mock import MockProvider
from verdict.verdict import Verdict


def _resolve_provider_from_env() -> BaseProvider:
    """Build a provider instance from VERDICT_PROVIDER environment variable.

    Supported values: mock, openai, anthropic, google, mistral, ollama, litellm.
    Falls back to MockProvider if unset.
    """
    provider_name = os.environ.get("VERDICT_PROVIDER", "mock").lower().strip()

    if provider_name == "mock":
        return MockProvider(response_fn=lambda prompt, msgs=None: prompt)

    # Lazy import to avoid requiring SDK packages at plugin load time
    if provider_name == "openai":
        from verdict.providers.openai import OpenAIProvider
        return OpenAIProvider()

    if provider_name == "anthropic":
        from verdict.providers.anthropic import AnthropicProvider
        return AnthropicProvider()

    if provider_name == "google":
        from verdict.providers.google import GoogleProvider
        return GoogleProvider()

    if provider_name == "mistral":
        from verdict.providers.mistral import MistralProvider
        return MistralProvider()

    if provider_name == "ollama":
        from verdict.providers.ollama import OllamaProvider
        return OllamaProvider()

    if provider_name == "litellm":
        from verdict.providers.litellm import LiteLLMProvider
        return LiteLLMProvider()

    raise ValueError(
        f"Unknown VERDICT_PROVIDER '{provider_name}'. "
        f"Supported: mock, openai, anthropic, google, mistral, ollama, litellm. "
        f"Set VERDICT_PROVIDER environment variable to one of these values."
    )


@pytest.fixture(scope="session")
def verdict_config() -> VerdictConfig:
    """Session-scoped Verdict configuration.

    Reads from environment or uses calibrated defaults.
    Override in conftest.py for project-specific settings.
    """
    return VerdictConfig()


@pytest.fixture(scope="session")
def verdict_provider() -> BaseProvider:
    """Session-scoped provider instance.

    Resolved from VERDICT_PROVIDER env var. Defaults to MockProvider
    when no provider is configured, so tests never fail due to
    missing credentials unless they explicitly require a real provider.
    """
    return _resolve_provider_from_env()


@pytest.fixture(scope="function")
def verdict_runner(verdict_provider: BaseProvider, verdict_config: VerdictConfig) -> Verdict:
    """Function-scoped Verdict instance.

    Fresh per test to prevent cross-test state leakage.
    Uses the session-scoped provider and config.
    """
    return Verdict(provider=verdict_provider, config=verdict_config)
