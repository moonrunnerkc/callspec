"""Pytest fixtures for Callspec integration.

callspec_config    (session scope)   - CallspecConfig loaded from env/conftest
callspec_provider  (session scope)   - configured provider instance
callspec_runner    (function scope)   - fresh Callspec instance per test
trajectory_runner    (function scope)   - calls provider and returns ToolCallTrajectory

These fixtures form the bridge between pytest and the Callspec API.
Developers use them in test files without manual instantiation.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from callspec.core.config import CallspecConfig
from callspec.core.trajectory import ToolCall, ToolCallTrajectory
from callspec.core.trajectory_builder import TrajectoryBuilder
from callspec.core.types import ProviderResponse
from callspec.providers.base import BaseProvider
from callspec.providers.mock import MockProvider
from callspec.verdict import Callspec


def _resolve_provider_from_env() -> BaseProvider:
    """Build a provider instance from CALLSPEC_PROVIDER environment variable.

    Supported values: mock, openai, anthropic, google, mistral, ollama, litellm.
    Falls back to MockProvider if unset.
    """
    provider_name = os.environ.get("CALLSPEC_PROVIDER", "mock").lower().strip()

    if provider_name == "mock":
        return MockProvider(response_fn=lambda prompt, msgs=None: prompt)

    # Lazy import to avoid requiring SDK packages at plugin load time
    if provider_name == "openai":
        from callspec.providers.openai import OpenAIProvider
        return OpenAIProvider()

    if provider_name == "anthropic":
        from callspec.providers.anthropic import AnthropicProvider
        return AnthropicProvider()

    if provider_name == "google":
        from callspec.providers.google import GoogleProvider
        return GoogleProvider()

    if provider_name == "mistral":
        from callspec.providers.mistral import MistralProvider
        return MistralProvider()

    if provider_name == "ollama":
        from callspec.providers.ollama import OllamaProvider
        return OllamaProvider()

    if provider_name == "litellm":
        from callspec.providers.litellm import LiteLLMProvider
        return LiteLLMProvider()

    raise ValueError(
        f"Unknown CALLSPEC_PROVIDER '{provider_name}'. "
        f"Supported: mock, openai, anthropic, google, mistral, ollama, litellm. "
        f"Set CALLSPEC_PROVIDER environment variable to one of these values."
    )


@pytest.fixture(scope="session")
def callspec_config() -> CallspecConfig:
    """Session-scoped Callspec configuration.

    Reads from environment or uses calibrated defaults.
    Override in conftest.py for project-specific settings.
    """
    return CallspecConfig()


@pytest.fixture(scope="session")
def callspec_provider() -> BaseProvider:
    """Session-scoped provider instance.

    Resolved from CALLSPEC_PROVIDER env var. Defaults to MockProvider
    when no provider is configured, so tests never fail due to
    missing credentials unless they explicitly require a real provider.
    """
    return _resolve_provider_from_env()


@pytest.fixture(scope="function")
def callspec_runner(
    callspec_provider: BaseProvider,
    callspec_config: CallspecConfig,
) -> Callspec:
    """Function-scoped Callspec instance.

    Fresh per test to prevent cross-test state leakage.
    Uses the session-scoped provider and config.
    """
    return Callspec(provider=callspec_provider, config=callspec_config)


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
    callspec_provider: BaseProvider,
    callspec_config: CallspecConfig,
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
    from callspec.core.runner import AssertionRunner

    runner = AssertionRunner(provider=callspec_provider, config=callspec_config)

    def _call_and_build(
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> TrajectoryBuilder:
        response = runner._call_provider_with_retries(prompt, messages)
        trajectory = _response_to_trajectory(response)
        return TrajectoryBuilder(trajectory=trajectory, config=callspec_config)

    return _call_and_build
