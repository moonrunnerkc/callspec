"""Integration tests for the Anthropic provider adapter.

Tests are split into:
1. Offline tests using a mocked Anthropic client (always run)
2. Live tests hitting the real API (skipped when ANTHROPIC_API_KEY is absent)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from callspec.core.types import ProviderResponse
from callspec.providers.anthropic import AnthropicProvider

# ---- Mock objects mirroring Anthropic SDK response shape ----

@dataclass
class MockTextBlock:
    type: str = "text"
    text: str = "The capital of France is Paris."


@dataclass
class MockAnthropicUsage:
    input_tokens: int = 15
    output_tokens: int = 9


@dataclass
class MockAnthropicResponse:
    id: str = "msg_01XFDUDYJgAACzvnptvVoYEL"
    type: str = "message"
    role: str = "assistant"
    content: list[MockTextBlock] = field(default_factory=lambda: [MockTextBlock()])
    model: str = "claude-sonnet-4-20250514"
    stop_reason: str = "end_turn"
    stop_sequence: str | None = None
    usage: MockAnthropicUsage = field(default_factory=MockAnthropicUsage)

    def model_dump(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role,
            "content": [{"type": b.type, "text": b.text} for b in self.content],
            "model": self.model,
            "stop_reason": self.stop_reason,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
        }


class TestAnthropicProviderOffline:

    def test_provider_name(self) -> None:
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")
        assert provider.provider_name == "anthropic"

    def test_import_error_gives_actionable_message(self) -> None:
        provider = AnthropicProvider()
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises(ImportError, match="pip install callspec\\[anthropic\\]"):
                provider._get_client()

    def test_response_normalization(self) -> None:
        provider = AnthropicProvider(model="claude-sonnet-4-20250514")

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()
        provider._client = mock_client

        response = provider.call("What is the capital of France?")

        assert isinstance(response, ProviderResponse)
        assert response.content == "The capital of France is Paris."
        assert response.model == "claude-sonnet-4-20250514"
        assert response.provider == "anthropic"
        assert response.prompt_tokens == 15
        assert response.completion_tokens == 9
        assert response.finish_reason == "end_turn"
        assert response.request_id == "msg_01XFDUDYJgAACzvnptvVoYEL"

    def test_system_message_extracted_separately(self) -> None:
        """Anthropic sends system as a top-level param, not in messages array."""
        provider = AnthropicProvider()

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()
        provider._client = mock_client

        messages = [
            {"role": "system", "content": "You are a geography expert."},
            {"role": "user", "content": "Capital of France?"},
        ]
        provider.call("fallback prompt", messages=messages)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["system"] == "You are a geography expert."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Capital of France?"}]

    def test_no_system_message(self) -> None:
        """Without a system message, none is passed to the API."""
        provider = AnthropicProvider()

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()
        provider._client = mock_client

        provider.call("What is 2+2?")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "system" not in call_kwargs
        assert call_kwargs["messages"] == [{"role": "user", "content": "What is 2+2?"}]

    def test_max_tokens_always_set(self) -> None:
        """Anthropic requires max_tokens on every request."""
        provider = AnthropicProvider(max_tokens=512)

        mock_client = MagicMock()
        mock_client.messages.create.return_value = MockAnthropicResponse()
        provider._client = mock_client

        provider.call("test")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 512

    def test_multiple_content_blocks_concatenated(self) -> None:
        """Multiple text blocks in the response are joined."""
        provider = AnthropicProvider()

        multi_block_response = MockAnthropicResponse(
            content=[
                MockTextBlock(text="First part. "),
                MockTextBlock(text="Second part."),
            ]
        )

        mock_client = MagicMock()
        mock_client.messages.create.return_value = multi_block_response
        provider._client = mock_client

        response = provider.call("test")
        assert response.content == "First part. Second part."

    def test_raw_response_preserved(self) -> None:
        provider = AnthropicProvider()

        mock_client = MagicMock()
        mock_response = MockAnthropicResponse()
        mock_client.messages.create.return_value = mock_response
        provider._client = mock_client

        response = provider.call("test")
        assert response.raw["id"] == "msg_01XFDUDYJgAACzvnptvVoYEL"
        assert response.raw["model"] == "claude-sonnet-4-20250514"

    @pytest.mark.asyncio
    async def test_async_call(self) -> None:
        import asyncio

        provider = AnthropicProvider()

        mock_async_client = MagicMock()
        future = asyncio.Future()
        future.set_result(MockAnthropicResponse())
        mock_async_client.messages.create.return_value = future

        provider._async_client = mock_async_client

        response = await provider.call_async("test")
        assert response.content == "The capital of France is Paris."
        assert response.provider == "anthropic"


# ---- Live tests ----

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set, skipping live Anthropic tests",
)
class TestAnthropicProviderLive:

    def test_live_call(self) -> None:
        provider = AnthropicProvider(
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=100,
        )
        response = provider.call("What is 2+2? Answer with just the number.")

        assert isinstance(response, ProviderResponse)
        assert "4" in response.content
        assert response.provider == "anthropic"
        assert response.prompt_tokens is not None
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_live_async_call(self) -> None:
        provider = AnthropicProvider(
            model="claude-sonnet-4-20250514",
            temperature=0.0,
            max_tokens=100,
        )
        response = await provider.call_async("What is 2+2? Answer with just the number.")

        assert isinstance(response, ProviderResponse)
        assert "4" in response.content
