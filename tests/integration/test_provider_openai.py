"""Integration tests for the OpenAI provider adapter.

Tests are split into two groups:
1. Offline tests using a mocked OpenAI client (always run)
2. Live tests hitting the real API (skipped when OPENAI_API_KEY is absent)

The offline tests validate response normalization, parameter building,
and error handling without API spend. The live tests confirm the adapter
works end-to-end against the real service.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.recorded_responses import OPENAI_CHAT_RESPONSE
from llm_assert.core.types import ProviderResponse
from llm_assert.providers.openai import OpenAIProvider

# ---- Mock objects that mirror the OpenAI SDK response shape ----

@dataclass
class MockUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class MockMessage:
    role: str
    content: str


@dataclass
class MockChoice:
    index: int
    message: MockMessage
    finish_reason: str


@dataclass
class MockCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: list[MockChoice]
    usage: MockUsage

    def model_dump(self) -> dict:
        return OPENAI_CHAT_RESPONSE


def _build_mock_completion() -> MockCompletion:
    return MockCompletion(
        id="chatcmpl-abc123def456",
        object="chat.completion",
        created=1700000000,
        model="gpt-4o-2024-11-20",
        choices=[
            MockChoice(
                index=0,
                message=MockMessage(role="assistant", content="The capital of France is Paris."),
                finish_reason="stop",
            )
        ],
        usage=MockUsage(prompt_tokens=12, completion_tokens=8, total_tokens=20),
    )


class TestOpenAIProviderOffline:
    """Offline tests using mocked OpenAI client."""

    def test_provider_name(self) -> None:
        provider = OpenAIProvider(model="gpt-4o")
        assert provider.provider_name == "openai"

    def test_import_error_gives_actionable_message(self) -> None:
        provider = OpenAIProvider(model="gpt-4o")
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="pip install llm-assert\\[openai\\]"):
                provider._get_client()

    def test_response_normalization(self) -> None:
        """Confirm provider response fields are correctly extracted."""
        provider = OpenAIProvider(model="gpt-4o", temperature=0.0, seed=42)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_completion()
        provider._client = mock_client

        response = provider.call("What is the capital of France?")

        assert isinstance(response, ProviderResponse)
        assert response.content == "The capital of France is Paris."
        # Extracts actual model from response, not the alias
        assert response.model == "gpt-4o-2024-11-20"
        assert response.provider == "openai"
        assert response.prompt_tokens == 12
        assert response.completion_tokens == 8
        assert response.finish_reason == "stop"
        assert response.request_id == "chatcmpl-abc123def456"
        assert response.latency_ms >= 0

    def test_messages_passed_through(self) -> None:
        """Custom messages array is forwarded to the API."""
        provider = OpenAIProvider(model="gpt-4o")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_completion()
        provider._client = mock_client

        custom_messages = [
            {"role": "system", "content": "You are a geography expert."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
        provider.call("ignored when messages provided", messages=custom_messages)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["messages"] == custom_messages

    def test_default_prompt_becomes_user_message(self) -> None:
        """When no messages provided, prompt becomes a user message."""
        provider = OpenAIProvider(model="gpt-4o")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_completion()
        provider._client = mock_client

        provider.call("What is the capital of France?")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["messages"] == [
            {"role": "user", "content": "What is the capital of France?"}
        ]

    def test_temperature_and_seed_passed(self) -> None:
        """Temperature and seed are forwarded to the API call."""
        provider = OpenAIProvider(model="gpt-4o", temperature=0.0, seed=42)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_completion()
        provider._client = mock_client

        provider.call("test")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0
        assert call_kwargs.kwargs["seed"] == 42

    def test_max_tokens_passed_when_set(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", max_tokens=100)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_completion()
        provider._client = mock_client

        provider.call("test")

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["max_tokens"] == 100

    def test_per_call_kwargs_override_defaults(self) -> None:
        provider = OpenAIProvider(model="gpt-4o", temperature=0.0)

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_completion()
        provider._client = mock_client

        provider.call("test", temperature=0.7)

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7

    def test_raw_response_preserved(self) -> None:
        """The raw field contains the full provider response dict."""
        provider = OpenAIProvider(model="gpt-4o")

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _build_mock_completion()
        provider._client = mock_client

        response = provider.call("test")
        assert response.raw == OPENAI_CHAT_RESPONSE

    def test_empty_content_handled(self) -> None:
        """Handles None content from the API gracefully."""
        provider = OpenAIProvider(model="gpt-4o")

        completion = _build_mock_completion()
        completion.choices[0].message.content = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = completion
        provider._client = mock_client

        response = provider.call("test")
        assert response.content == ""

    @pytest.mark.asyncio
    async def test_async_call(self) -> None:
        """Async call uses AsyncOpenAI and returns normalized response."""
        provider = OpenAIProvider(model="gpt-4o")

        mock_async_client = MagicMock()
        mock_completion = _build_mock_completion()
        mock_async_client.chat.completions.create = MagicMock(return_value=mock_completion)

        # Make the mock awaitable
        import asyncio
        future = asyncio.Future()
        future.set_result(mock_completion)
        mock_async_client.chat.completions.create.return_value = future

        provider._async_client = mock_async_client

        response = await provider.call_async("test")
        assert response.content == "The capital of France is Paris."
        assert response.provider == "openai"


# ---- Live tests (skipped without API key) ----

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set, skipping live OpenAI tests",
)
class TestOpenAIProviderLive:

    def test_live_call(self) -> None:
        provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.0, seed=42)
        response = provider.call("What is 2+2? Answer with just the number.")

        assert isinstance(response, ProviderResponse)
        assert "4" in response.content
        assert response.provider == "openai"
        assert response.prompt_tokens is not None
        assert response.completion_tokens is not None
        assert response.latency_ms > 0
        # Model field should be the actual model, not alias
        assert "gpt-4o-mini" in response.model

    @pytest.mark.asyncio
    async def test_live_async_call(self) -> None:
        provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.0, seed=42)
        response = await provider.call_async("What is 2+2? Answer with just the number.")

        assert isinstance(response, ProviderResponse)
        assert "4" in response.content
