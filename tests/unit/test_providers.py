"""Unit tests for MockProvider and provider response normalization."""

from __future__ import annotations

from llm_assert.core.types import ProviderResponse
from llm_assert.providers.mock import MockProvider


class TestMockProvider:

    def test_basic_call(self) -> None:
        provider = MockProvider(lambda prompt, messages: f"echo: {prompt}")
        response = provider.call("hello")
        assert response.content == "echo: hello"
        assert response.provider == "mock"
        assert response.model == "mock"
        assert response.finish_reason == "stop"

    def test_custom_model_name(self) -> None:
        provider = MockProvider(
            lambda prompt, messages: "response",
            model_name="custom-mock-v1",
        )
        response = provider.call("test")
        assert response.model == "custom-mock-v1"

    def test_messages_passed_through(self) -> None:
        captured = {}

        def capture_fn(prompt: str, messages: list) -> str:
            captured["prompt"] = prompt
            captured["messages"] = messages
            return "captured"

        provider = MockProvider(capture_fn)
        messages = [{"role": "user", "content": "hi"}]
        response = provider.call("system prompt", messages=messages)
        assert response.content == "captured"
        assert captured["messages"] == messages

    def test_raw_response_contains_input(self) -> None:
        provider = MockProvider(lambda prompt, messages: "output")
        response = provider.call("input prompt")
        assert response.raw["prompt"] == "input prompt"

    def test_latency_override(self) -> None:
        provider = MockProvider(lambda prompt, messages: "fast", latency_ms=100)
        response = provider.call("test")
        # Mock latency adds 100ms to the (near-zero) actual call time
        assert response.latency_ms >= 100

    def test_provider_name(self) -> None:
        provider = MockProvider(lambda prompt, messages: "x")
        assert provider.provider_name == "mock"


class TestProviderResponse:

    def test_frozen_dataclass(self) -> None:
        response = ProviderResponse(content="test")
        assert response.content == "test"
        assert response.model == "unknown"
        assert response.provider == "unknown"
        assert response.prompt_tokens is None
        assert response.completion_tokens is None

    def test_all_fields(self) -> None:
        response = ProviderResponse(
            content="hello",
            raw={"original": True},
            model="gpt-4o-2024-11-20",
            provider="openai",
            latency_ms=150,
            prompt_tokens=10,
            completion_tokens=20,
            finish_reason="stop",
            request_id="req-abc123",
        )
        assert response.content == "hello"
        assert response.model == "gpt-4o-2024-11-20"
        assert response.latency_ms == 150
        assert response.request_id == "req-abc123"
