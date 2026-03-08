"""Pytest fixtures for LLMAssert integration.

llm_assert_config  (session scope) - LLMAssertConfig loaded from env/conftest
llm_assert_provider (session scope) - configured provider instance
llm_assert_runner  (function scope) - fresh LLMAssert instance per test

These fixtures form the bridge between pytest and the LLMAssert API.
Developers use them in test files without manual instantiation.
"""

from __future__ import annotations

import os

import pytest

from llm_assert.core.config import LLMAssertConfig
from llm_assert.providers.base import BaseProvider
from llm_assert.providers.mock import MockProvider
from llm_assert.verdict import LLMAssert


def _resolve_provider_from_env() -> BaseProvider:
    """Build a provider instance from LLM_ASSERT_PROVIDER environment variable.

    Supported values: mock, openai, anthropic, google, mistral, ollama, litellm.
    Falls back to MockProvider if unset.
    """
    provider_name = os.environ.get("LLM_ASSERT_PROVIDER", "mock").lower().strip()

    if provider_name == "mock":
        return MockProvider(response_fn=lambda prompt, msgs=None: prompt)

    # Lazy import to avoid requiring SDK packages at plugin load time
    if provider_name == "openai":
        from llm_assert.providers.openai import OpenAIProvider
        return OpenAIProvider()

    if provider_name == "anthropic":
        from llm_assert.providers.anthropic import AnthropicProvider
        return AnthropicProvider()

    if provider_name == "google":
        from llm_assert.providers.google import GoogleProvider
        return GoogleProvider()

    if provider_name == "mistral":
        from llm_assert.providers.mistral import MistralProvider
        return MistralProvider()

    if provider_name == "ollama":
        from llm_assert.providers.ollama import OllamaProvider
        return OllamaProvider()

    if provider_name == "litellm":
        from llm_assert.providers.litellm import LiteLLMProvider
        return LiteLLMProvider()

    raise ValueError(
        f"Unknown LLM_ASSERT_PROVIDER '{provider_name}'. "
        f"Supported: mock, openai, anthropic, google, mistral, ollama, litellm. "
        f"Set LLM_ASSERT_PROVIDER environment variable to one of these values."
    )


@pytest.fixture(scope="session")
def llm_assert_config() -> LLMAssertConfig:
    """Session-scoped LLMAssert configuration.

    Reads from environment or uses calibrated defaults.
    Override in conftest.py for project-specific settings.
    """
    return LLMAssertConfig()


@pytest.fixture(scope="session")
def llm_assert_provider() -> BaseProvider:
    """Session-scoped provider instance.

    Resolved from LLM_ASSERT_PROVIDER env var. Defaults to MockProvider
    when no provider is configured, so tests never fail due to
    missing credentials unless they explicitly require a real provider.
    """
    return _resolve_provider_from_env()


@pytest.fixture(scope="function")
def llm_assert_runner(
    llm_assert_provider: BaseProvider,
    llm_assert_config: LLMAssertConfig,
) -> LLMAssert:
    """Function-scoped LLMAssert instance.

    Fresh per test to prevent cross-test state leakage.
    Uses the session-scoped provider and config.
    """
    return LLMAssert(provider=llm_assert_provider, config=llm_assert_config)
