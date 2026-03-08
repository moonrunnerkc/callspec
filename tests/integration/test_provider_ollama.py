"""Integration tests for the Ollama provider adapter.

Tests are split into:
1. Offline tests using a mocked Ollama client (always run)
2. Live tests hitting a local Ollama server (skipped when unavailable)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from llm_assert.core.types import ProviderResponse
from llm_assert.providers.ollama import OllamaProvider
from tests.fixtures.recorded_responses import OLLAMA_CHAT_RESPONSE


class TestOllamaProviderOffline:

    def test_provider_name(self) -> None:
        provider = OllamaProvider(model="llama3")
        assert provider.provider_name == "ollama"

    def test_import_error_gives_actionable_message(self) -> None:
        provider = OllamaProvider()
        with patch.dict("sys.modules", {"ollama": None}):
            with pytest.raises(ImportError, match="pip install llm-assert\\[ollama\\]"):
                provider._get_client()

    def test_response_normalization(self) -> None:
        provider = OllamaProvider(model="llama3")

        mock_client = MagicMock()
        mock_client.chat.return_value = OLLAMA_CHAT_RESPONSE
        provider._client = mock_client

        response = provider.call("What is the capital of France?")

        assert isinstance(response, ProviderResponse)
        assert response.content == "The capital of France is Paris."
        assert response.model == "llama3:latest"
        assert response.provider == "ollama"
        assert response.prompt_tokens == 14
        assert response.completion_tokens == 8
        assert response.finish_reason == "stop"
        assert response.request_id is None
        assert response.latency_ms >= 0

    def test_messages_forwarded(self) -> None:
        provider = OllamaProvider(model="llama3")

        mock_client = MagicMock()
        mock_client.chat.return_value = OLLAMA_CHAT_RESPONSE
        provider._client = mock_client

        custom_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Capital?"},
        ]
        provider.call("ignored", messages=custom_messages)

        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs["messages"] == custom_messages

    def test_temperature_and_seed_in_options(self) -> None:
        provider = OllamaProvider(model="llama3", temperature=0.0, seed=42)

        mock_client = MagicMock()
        mock_client.chat.return_value = OLLAMA_CHAT_RESPONSE
        provider._client = mock_client

        provider.call("test")

        call_kwargs = mock_client.chat.call_args
        options = call_kwargs.kwargs["options"]
        assert options["temperature"] == 0.0
        assert options["seed"] == 42

    def test_num_predict_passed_when_set(self) -> None:
        provider = OllamaProvider(model="llama3", num_predict=100)

        mock_client = MagicMock()
        mock_client.chat.return_value = OLLAMA_CHAT_RESPONSE
        provider._client = mock_client

        provider.call("test")

        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs["options"]["num_predict"] == 100

    def test_missing_token_counts_handled(self) -> None:
        """Ollama may not return token counts depending on version."""
        response_without_tokens = {
            "model": "llama3:latest",
            "message": {"role": "assistant", "content": "Hello"},
            "done": True,
        }

        provider = OllamaProvider(model="llama3")
        mock_client = MagicMock()
        mock_client.chat.return_value = response_without_tokens
        provider._client = mock_client

        response = provider.call("test")
        assert response.prompt_tokens is None
        assert response.completion_tokens is None
        assert response.finish_reason == "stop"

    def test_raw_response_preserved(self) -> None:
        provider = OllamaProvider(model="llama3")

        mock_client = MagicMock()
        mock_client.chat.return_value = OLLAMA_CHAT_RESPONSE
        provider._client = mock_client

        response = provider.call("test")
        assert response.raw["model"] == "llama3:latest"
        assert response.raw["done"] is True

    @pytest.mark.asyncio
    async def test_async_call(self) -> None:
        import asyncio

        provider = OllamaProvider(model="llama3")

        mock_async_client = MagicMock()
        future = asyncio.Future()
        future.set_result(OLLAMA_CHAT_RESPONSE)
        mock_async_client.chat.return_value = future

        provider._async_client = mock_async_client

        response = await provider.call_async("test")
        assert response.content == "The capital of France is Paris."
        assert response.provider == "ollama"


# ---- Live tests (require local Ollama server) ----

@pytest.mark.skipif(
    not os.environ.get("LLM_ASSERT_OLLAMA_LIVE"),
    reason="LLM_ASSERT_OLLAMA_LIVE not set, skipping live Ollama tests",
)
class TestOllamaProviderLive:

    def test_live_call(self) -> None:
        provider = OllamaProvider(model="llama3", temperature=0.0, seed=42)
        response = provider.call("What is 2+2? Answer with just the number.")

        assert isinstance(response, ProviderResponse)
        assert "4" in response.content
        assert response.provider == "ollama"
        assert response.latency_ms > 0
