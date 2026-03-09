"""Pytest fixtures for LLMAssert integration.

llm_assert_config    (session scope)   - LLMAssertConfig loaded from env/conftest
llm_assert_provider  (session scope)   - configured provider instance
llm_assert_runner    (function scope)   - fresh LLMAssert instance per test
trajectory_runner    (function scope)   - calls provider and returns ToolCallTrajectory

These fixtures form the bridge between pytest and the LLMAssert API.
Developers use them in test files without manual instantiation.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from llm_assert.core.config import LLMAssertConfig
from llm_assert.core.trajectory import ToolCall, ToolCallTrajectory
from llm_assert.core.trajectory_builder import TrajectoryBuilder
from llm_assert.core.types import ProviderResponse
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


def _response_to_trajectory(response: ProviderResponse) -> ToolCallTrajectory:
    """Convert a ProviderResponse with tool_calls into a ToolCallTrajectory."""
    calls = [
        ToolCall(
            tool_name=tc.get("name", tc.get("tool_name", "unknown")),
            arguments=tc.get("arguments", {}),
            call_index=i,
        )
        for i, tc in enumerate(response.tool_calls)
    ]
    return ToolCallTrajectory(
        calls=calls,
        model=response.model,
        provider=response.provider,
        raw_response=response.raw,
    )


@pytest.fixture(scope="function")
def trajectory_runner(
    llm_assert_provider: BaseProvider,
    llm_assert_config: LLMAssertConfig,
):
    """Function-scoped factory that calls the provider and returns a TrajectoryBuilder.

    Usage in a test:
        def test_agent_flow(trajectory_runner):
            builder = trajectory_runner("Book a flight from NYC to London")
            result = (
                builder
                .calls_tools_in_order(["search_flights", "book_flight"])
                .argument_not_empty("search_flights", "origin")
                .run()
            )
            assert result.passed

    The fixture calls the provider once, extracts tool calls from the response,
    builds a ToolCallTrajectory, and returns a ready-to-chain TrajectoryBuilder.
    """
    from llm_assert.core.runner import AssertionRunner

    runner = AssertionRunner(provider=llm_assert_provider, config=llm_assert_config)

    def _call_and_build(
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> TrajectoryBuilder:
        response = runner._call_provider_with_retries(prompt, messages)
        trajectory = _response_to_trajectory(response)
        return TrajectoryBuilder(trajectory=trajectory, config=llm_assert_config)

    return _call_and_build
