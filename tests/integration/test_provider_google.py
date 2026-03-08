"""Integration tests for the Google provider adapter.

Tests are split into:
1. Offline tests using mocked google-generativeai objects (always run)
2. Live tests hitting the real API (skipped when GOOGLE_API_KEY is absent)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest

from llm_assert.core.types import ProviderResponse
from llm_assert.providers.google import GoogleProvider

# ---- Mock objects mirroring google-generativeai response shape ----

@dataclass
class MockPart:
    text: str = "The capital of France is Paris."


@dataclass
class MockContent:
    parts: list[MockPart] = field(default_factory=lambda: [MockPart()])
    role: str = "model"


@dataclass
class MockCandidate:
    content: MockContent = field(default_factory=MockContent)
    finish_reason: str = "STOP"
    safety_ratings: list = field(default_factory=list)


@dataclass
class MockUsageMetadata:
    prompt_token_count: int = 10
    candidates_token_count: int = 8
    total_token_count: int = 18


@dataclass
class MockGenerateResponse:
    text: str = "The capital of France is Paris."
    candidates: list[MockCandidate] = field(default_factory=lambda: [MockCandidate()])
    usage_metadata: MockUsageMetadata = field(default_factory=MockUsageMetadata)


class TestGoogleProviderOffline:

    def test_provider_name(self) -> None:
        provider = GoogleProvider(model="gemini-2.0-flash")
        assert provider.provider_name == "google"

    def test_import_error_gives_actionable_message(self) -> None:
        provider = GoogleProvider()
        with patch.dict("sys.modules", {"google": None, "google.generativeai": None}):
            with pytest.raises(ImportError, match="pip install llm-assert\\[google\\]"):
                provider._configure_sdk()

    def test_response_normalization(self) -> None:
        provider = GoogleProvider(model="gemini-2.0-flash")

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MockGenerateResponse()
        provider._model = mock_model

        response = provider.call("What is the capital of France?")

        assert isinstance(response, ProviderResponse)
        assert response.content == "The capital of France is Paris."
        assert response.model == "gemini-2.0-flash"
        assert response.provider == "google"
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 8
        assert response.finish_reason == "STOP"
        assert response.request_id is None

    def test_prompt_as_simple_string(self) -> None:
        """When no messages provided, prompt is passed as a simple string."""
        provider = GoogleProvider(model="gemini-2.0-flash")

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MockGenerateResponse()
        provider._model = mock_model

        provider.call("Hello")

        call_args = mock_model.generate_content.call_args
        assert call_args.args[0] == ["Hello"]

    def test_messages_converted_to_google_format(self) -> None:
        """Assistant role maps to model role for Google's API."""
        provider = GoogleProvider(model="gemini-2.0-flash")

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MockGenerateResponse()
        provider._model = mock_model

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Capital of France?"},
        ]
        provider.call("ignored", messages=messages)

        call_args = mock_model.generate_content.call_args
        contents = call_args.args[0]
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"  # assistant -> model
        assert contents[2]["role"] == "user"

    def test_system_messages_skipped(self) -> None:
        """System messages are skipped in contents (handled at model level)."""
        provider = GoogleProvider(model="gemini-2.0-flash")

        mock_model = MagicMock()
        mock_model.generate_content.return_value = MockGenerateResponse()
        provider._model = mock_model

        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]
        provider.call("fallback", messages=messages)

        call_args = mock_model.generate_content.call_args
        contents = call_args.args[0]
        # System message should not appear in contents
        assert len(contents) == 1
        assert contents[0]["role"] == "user"

    def test_empty_text_handled(self) -> None:
        provider = GoogleProvider(model="gemini-2.0-flash")

        empty_response = MockGenerateResponse(text="")

        mock_model = MagicMock()
        mock_model.generate_content.return_value = empty_response
        provider._model = mock_model

        response = provider.call("test")
        assert response.content == ""

    def test_missing_usage_metadata_handled(self) -> None:
        provider = GoogleProvider(model="gemini-2.0-flash")

        no_usage = MockGenerateResponse()
        no_usage.usage_metadata = None

        mock_model = MagicMock()
        mock_model.generate_content.return_value = no_usage
        provider._model = mock_model

        response = provider.call("test")
        assert response.prompt_tokens is None
        assert response.completion_tokens is None

    @pytest.mark.asyncio
    async def test_async_call(self) -> None:
        import asyncio

        provider = GoogleProvider(model="gemini-2.0-flash")

        mock_model = MagicMock()
        future = asyncio.Future()
        future.set_result(MockGenerateResponse())
        mock_model.generate_content_async.return_value = future
        provider._model = mock_model

        response = await provider.call_async("test")
        assert response.content == "The capital of France is Paris."
        assert response.provider == "google"


# ---- Live tests ----

@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set, skipping live Google tests",
)
class TestGoogleProviderLive:

    def test_live_call(self) -> None:
        provider = GoogleProvider(model="gemini-2.0-flash", temperature=0.0)
        response = provider.call("What is 2+2? Answer with just the number.")

        assert isinstance(response, ProviderResponse)
        assert "4" in response.content
        assert response.provider == "google"
        assert response.latency_ms > 0
