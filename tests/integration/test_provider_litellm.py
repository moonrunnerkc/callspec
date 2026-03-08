"""Integration tests for the LiteLLM provider adapter.

Tests are split into:
1. Offline tests using a mocked litellm module (always run)
2. Live tests hitting the real API via LiteLLM (skipped when no API key)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from llm_assert.core.types import ProviderResponse
from llm_assert.providers.litellm import LiteLLMProvider
from tests.fixtures.recorded_responses import LITELLM_COMPLETION_RESPONSE

# ---- Mock objects mirroring LiteLLM's OpenAI-compatible response shape ----

@dataclass
class MockLiteLLMUsage:
    prompt_tokens: int = 12
    completion_tokens: int = 8
    total_tokens: int = 20


@dataclass
class MockLiteLLMMessage:
    role: str = "assistant"
    content: str = "The capital of France is Paris."


@dataclass
class MockLiteLLMChoice:
    index: int = 0
    message: MockLiteLLMMessage = field(default_factory=MockLiteLLMMessage)
    finish_reason: str = "stop"


@dataclass
class MockLiteLLMResponse:
    id: str = "chatcmpl-litellm-xyz789"
    object: str = "chat.completion"
    created: int = 1700000000
    model: str = "gpt-4o-2024-11-20"
    choices: list[MockLiteLLMChoice] = field(default_factory=lambda: [MockLiteLLMChoice()])
    usage: MockLiteLLMUsage = field(default_factory=MockLiteLLMUsage)

    def model_dump(self) -> dict:
        return LITELLM_COMPLETION_RESPONSE


class TestLiteLLMProviderOffline:

    def test_provider_name(self) -> None:
        provider = LiteLLMProvider(model="gpt-4o")
        assert provider.provider_name == "litellm"

    def test_import_error_gives_actionable_message(self) -> None:
        provider = LiteLLMProvider()
        with patch.dict("sys.modules", {"litellm": None}):
            with pytest.raises(ImportError, match="pip install llm-assert\\[litellm\\]"):
                provider._get_litellm()

    def test_response_normalization(self) -> None:
        provider = LiteLLMProvider(model="gpt-4o")

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MockLiteLLMResponse()
        provider._litellm = mock_litellm

        response = provider.call("What is the capital of France?")

        assert isinstance(response, ProviderResponse)
        assert response.content == "The capital of France is Paris."
        assert response.model == "gpt-4o-2024-11-20"
        assert response.provider == "litellm"
        assert response.prompt_tokens == 12
        assert response.completion_tokens == 8
        assert response.finish_reason == "stop"
        assert response.request_id == "chatcmpl-litellm-xyz789"

    def test_messages_forwarded(self) -> None:
        provider = LiteLLMProvider(model="gpt-4o")

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MockLiteLLMResponse()
        provider._litellm = mock_litellm

        custom_messages = [{"role": "user", "content": "Hi"}]
        provider.call("ignored", messages=custom_messages)

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["messages"] == custom_messages

    def test_api_key_forwarded(self) -> None:
        provider = LiteLLMProvider(model="gpt-4o", api_key="sk-test123")

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MockLiteLLMResponse()
        provider._litellm = mock_litellm

        provider.call("test")

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["api_key"] == "sk-test123"

    def test_api_base_forwarded(self) -> None:
        provider = LiteLLMProvider(model="gpt-4o", api_base="https://custom.api.com")

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MockLiteLLMResponse()
        provider._litellm = mock_litellm

        provider.call("test")

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["api_base"] == "https://custom.api.com"

    def test_temperature_and_seed_passed(self) -> None:
        provider = LiteLLMProvider(model="gpt-4o", temperature=0.0, seed=42)

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MockLiteLLMResponse()
        provider._litellm = mock_litellm

        provider.call("test")

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0
        assert call_kwargs.kwargs["seed"] == 42

    def test_provider_name_extraction_from_model_string(self) -> None:
        """LiteLLM model strings with / contain the provider prefix."""
        provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-20250514")
        assert provider._extract_provider_from_model() == "anthropic"

        provider_no_prefix = LiteLLMProvider(model="gpt-4o")
        assert provider_no_prefix._extract_provider_from_model() == "litellm"

    def test_raw_response_preserved(self) -> None:
        provider = LiteLLMProvider(model="gpt-4o")

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MockLiteLLMResponse()
        provider._litellm = mock_litellm

        response = provider.call("test")
        assert response.raw == LITELLM_COMPLETION_RESPONSE

    def test_per_call_kwargs_override(self) -> None:
        provider = LiteLLMProvider(model="gpt-4o", temperature=0.0)

        mock_litellm = MagicMock()
        mock_litellm.completion.return_value = MockLiteLLMResponse()
        provider._litellm = mock_litellm

        provider.call("test", temperature=0.8)

        call_kwargs = mock_litellm.completion.call_args
        assert call_kwargs.kwargs["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_async_call(self) -> None:
        import asyncio

        provider = LiteLLMProvider(model="gpt-4o")

        mock_litellm = MagicMock()
        future = asyncio.Future()
        future.set_result(MockLiteLLMResponse())
        mock_litellm.acompletion.return_value = future
        provider._litellm = mock_litellm

        response = await provider.call_async("test")
        assert response.content == "The capital of France is Paris."
        assert response.provider == "litellm"


# ---- Live tests (requires OPENAI_API_KEY since default model is gpt-4o) ----

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set, skipping live LiteLLM tests",
)
class TestLiteLLMProviderLive:

    def test_live_call(self) -> None:
        provider = LiteLLMProvider(
            model="gpt-4o-mini",
            temperature=0.0,
            seed=42,
            max_tokens=50,
        )
        response = provider.call("What is 2+2? Answer with just the number.")

        assert isinstance(response, ProviderResponse)
        assert "4" in response.content
        assert response.provider == "litellm"
        assert response.latency_ms > 0
