"""Tests for CaptureInterceptor: wraps provider calls with trajectory capture.

Uses MockProvider with tool_calls_fn to simulate provider responses
containing tool calls. Verifies capture(), call_and_capture(), and edge
cases like empty tool calls.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_assert.capture.interceptor import CaptureInterceptor
from llm_assert.core.trajectory import ToolCallTrajectory
from llm_assert.core.types import ProviderResponse
from llm_assert.providers.mock import MockProvider


def _echo_response(prompt: str, messages: Any = None) -> str:
    return f"Response to: {prompt}"


def _tool_calls_for_search(prompt: str, messages: Any = None) -> list[dict[str, Any]]:
    """Mock tool_calls_fn that always returns a search + book sequence."""
    return [
        {"name": "search", "arguments": {"q": prompt}, "id": "tc_1"},
        {"name": "book", "arguments": {"id": "BA123"}, "id": "tc_2"},
    ]


def _no_tool_calls(prompt: str, messages: Any = None) -> list[dict[str, Any]]:
    return []


class TestCaptureInterceptor:
    """CaptureInterceptor: provider call + trajectory capture in one step."""

    def test_capture_returns_trajectory(self):
        provider = MockProvider(
            response_fn=_echo_response,
            model_name="gpt-4o-mock",
            tool_calls_fn=_tool_calls_for_search,
        )
        interceptor = CaptureInterceptor(provider)
        traj = interceptor.capture("find flights to London")

        assert isinstance(traj, ToolCallTrajectory)
        assert len(traj) == 2
        assert traj.tool_names == ["search", "book"]
        assert traj.model == "gpt-4o-mock"
        assert traj.provider == "mock"

    def test_capture_with_no_tool_calls(self):
        provider = MockProvider(
            response_fn=_echo_response,
            tool_calls_fn=_no_tool_calls,
        )
        interceptor = CaptureInterceptor(provider)
        traj = interceptor.capture("hello")
        assert traj.is_empty

    def test_capture_without_tool_calls_fn(self):
        """Provider with no tool_calls_fn returns empty trajectory."""
        provider = MockProvider(response_fn=_echo_response)
        interceptor = CaptureInterceptor(provider)
        traj = interceptor.capture("hello")
        assert traj.is_empty

    def test_call_and_capture_returns_both(self):
        provider = MockProvider(
            response_fn=_echo_response,
            model_name="test-model",
            tool_calls_fn=_tool_calls_for_search,
        )
        interceptor = CaptureInterceptor(provider)
        response, traj = interceptor.call_and_capture("book a flight")

        # Verify the response is a ProviderResponse
        assert isinstance(response, ProviderResponse)
        assert response.content == "Response to: book a flight"
        assert response.model == "test-model"

        # Verify the trajectory matches
        assert isinstance(traj, ToolCallTrajectory)
        assert len(traj) == 2
        assert traj.tool_names == ["search", "book"]

    def test_capture_passes_arguments_through(self):
        """Verify prompt and messages reach the provider."""
        captured_args: dict[str, Any] = {}

        def recording_fn(prompt: str, messages: Any = None) -> str:
            captured_args["prompt"] = prompt
            captured_args["messages"] = messages
            return "ok"

        provider = MockProvider(response_fn=recording_fn)
        interceptor = CaptureInterceptor(provider)

        messages = [{"role": "user", "content": "test"}]
        interceptor.capture("system prompt", messages=messages)

        assert captured_args["prompt"] == "system prompt"
        assert captured_args["messages"] == messages

    def test_trajectory_arguments_reflect_prompt(self):
        """The mock tool_calls_fn uses the prompt in the search query."""
        provider = MockProvider(
            response_fn=_echo_response,
            tool_calls_fn=_tool_calls_for_search,
        )
        interceptor = CaptureInterceptor(provider)
        traj = interceptor.capture("Paris hotels")
        assert traj.calls[0].arguments["q"] == "Paris hotels"


@pytest.mark.asyncio
class TestCaptureInterceptorAsync:
    """Async variant of CaptureInterceptor."""

    async def test_capture_async(self):
        provider = MockProvider(
            response_fn=_echo_response,
            model_name="async-model",
            tool_calls_fn=_tool_calls_for_search,
        )
        interceptor = CaptureInterceptor(provider)
        traj = await interceptor.capture_async("find flights")

        assert isinstance(traj, ToolCallTrajectory)
        assert len(traj) == 2
        assert traj.model == "async-model"
