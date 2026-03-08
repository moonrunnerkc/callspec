"""Integration tests for the Mistral provider adapter.

Tests are split into:
1. Offline tests using a mocked Mistral client (always run)
2. Live tests hitting the real API (skipped when MISTRAL_API_KEY is absent)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from tests.fixtures.recorded_responses import MISTRAL_CHAT_RESPONSE
from verdict.core.types import ProviderResponse
from verdict.providers.mistral import MistralProvider

# ---- Mock objects mirroring Mistral SDK response shape ----

@dataclass
class MockMistralUsage:
    prompt_tokens: int = 11
    completion_tokens: int = 8
    total_tokens: int = 19


@dataclass
class MockMistralMessage:
    role: str = "assistant"
    content: str = "The capital of France is Paris."


@dataclass
class MockMistralChoice:
    index: int = 0
    message: MockMistralMessage = field(default_factory=MockMistralMessage)
    finish_reason: str = "stop"


@dataclass
class MockMistralResponse:
    id: str = "cmpl-e5cc70bb28c34"
    object: str = "chat.completion"
    created: int = 1700000000
    model: str = "mistral-large-latest"
    choices: list[MockMistralChoice] = field(default_factory=lambda: [MockMistralChoice()])
    usage: MockMistralUsage = field(default_factory=MockMistralUsage)

    def model_dump(self) -> dict:
        return MISTRAL_CHAT_RESPONSE


class TestMistralProviderOffline:

    def test_provider_name(self) -> None:
        provider = MistralProvider(model="mistral-large-latest")
        assert provider.provider_name == "mistral"

    def test_import_error_gives_actionable_message(self) -> None:
        provider = MistralProvider()
        with patch.dict("sys.modules", {"mistralai": None}):
            with pytest.raises(ImportError, match="pip install verdict\\[mistral\\]"):
                provider._get_client()

    def test_response_normalization(self) -> None:
        provider = MistralProvider(model="mistral-large-latest")

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MockMistralResponse()
        provider._client = mock_client

        response = provider.call("What is the capital of France?")

        assert isinstance(response, ProviderResponse)
        assert response.content == "The capital of France is Paris."
        assert response.model == "mistral-large-latest"
        assert response.provider == "mistral"
        assert response.prompt_tokens == 11
        assert response.completion_tokens == 8
        assert response.finish_reason == "stop"
        assert response.request_id == "cmpl-e5cc70bb28c34"

    def test_messages_forwarded(self) -> None:
        provider = MistralProvider(model="mistral-large-latest")

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MockMistralResponse()
        provider._client = mock_client

        custom_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Capital?"},
        ]
        provider.call("ignored", messages=custom_messages)

        call_kwargs = mock_client.chat.complete.call_args
        assert call_kwargs.kwargs["messages"] == custom_messages

    def test_temperature_passed(self) -> None:
        provider = MistralProvider(model="mistral-large-latest", temperature=0.0)

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MockMistralResponse()
        provider._client = mock_client

        provider.call("test")

        call_kwargs = mock_client.chat.complete.call_args
        assert call_kwargs.kwargs["temperature"] == 0.0

    def test_max_tokens_passed_when_set(self) -> None:
        provider = MistralProvider(model="mistral-large-latest", max_tokens=200)

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MockMistralResponse()
        provider._client = mock_client

        provider.call("test")

        call_kwargs = mock_client.chat.complete.call_args
        assert call_kwargs.kwargs["max_tokens"] == 200

    def test_raw_response_preserved(self) -> None:
        provider = MistralProvider(model="mistral-large-latest")

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = MockMistralResponse()
        provider._client = mock_client

        response = provider.call("test")
        assert response.raw == MISTRAL_CHAT_RESPONSE

    def test_empty_content_handled(self) -> None:
        provider = MistralProvider()

        mock_response = MockMistralResponse()
        mock_response.choices[0].message.content = None

        mock_client = MagicMock()
        mock_client.chat.complete.return_value = mock_response
        provider._client = mock_client

        response = provider.call("test")
        assert response.content == ""

    @pytest.mark.asyncio
    async def test_async_call(self) -> None:
        import asyncio

        provider = MistralProvider()

        mock_client = MagicMock()
        future = asyncio.Future()
        future.set_result(MockMistralResponse())
        mock_client.chat.complete_async.return_value = future
        provider._client = mock_client

        response = await provider.call_async("test")
        assert response.content == "The capital of France is Paris."
        assert response.provider == "mistral"


# ---- Live tests ----

@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set, skipping live Mistral tests",
)
class TestMistralProviderLive:

    def test_live_call(self) -> None:
        provider = MistralProvider(
            model="mistral-small-latest",
            temperature=0.0,
            max_tokens=100,
        )
        response = provider.call("What is 2+2? Answer with just the number.")

        assert isinstance(response, ProviderResponse)
        assert "4" in response.content
        assert response.provider == "mistral"
        assert response.latency_ms > 0
