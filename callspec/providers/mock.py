"""MockProvider: deterministic function-based provider for testing.

No network calls. Takes a function that maps (prompt, messages) to a string
and wraps the output in a ProviderResponse with stable mock metadata.
Used in Callspec's own test suite and by teams testing assertion configurations
without spending API credits.

An optional tool_calls_fn returns tool call dicts for testing tool-call
capture and contract assertions without real provider calls.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from callspec.core.types import ProviderResponse
from callspec.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Provider that returns deterministic responses from a user-supplied function.

    The response function receives (prompt, messages) and returns a string.
    That string becomes the `content` of a ProviderResponse with zeroed-out
    latency and token counts.

    The optional tool_calls_fn receives (prompt, messages) and returns a list
    of tool call dicts (each with at minimum "name" and "arguments" keys).
    This enables testing tool-call assertions without API spend.
    """

    def __init__(
        self,
        response_fn: Callable[..., str],
        model_name: str = "mock",
        latency_ms: int = 0,
        tool_calls_fn: Callable[..., list[dict[str, Any]]] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self._response_fn = response_fn
        self._model_name = model_name
        self._latency_ms = latency_ms

        # Accept either a callable or a static list for tool calls.
        # Static list is wrapped into a constant function for convenience.
        if tool_calls is not None and tool_calls_fn is None:
            static_calls = list(tool_calls)
            self._tool_calls_fn: Callable[
                ..., list[dict[str, Any]]
            ] | None = lambda p, m=None: static_calls
        else:
            self._tool_calls_fn = tool_calls_fn

    @property
    def provider_name(self) -> str:
        return "mock"

    def call(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        start = time.monotonic()
        content = self._response_fn(prompt, messages)
        elapsed_ms = int((time.monotonic() - start) * 1000) + self._latency_ms

        tool_calls: list[dict[str, Any]] = []
        if self._tool_calls_fn is not None:
            tool_calls = self._tool_calls_fn(prompt, messages)

        return ProviderResponse(
            content=content,
            raw={"prompt": prompt, "messages": messages},
            model=self._model_name,
            provider="mock",
            latency_ms=elapsed_ms,
            prompt_tokens=None,
            completion_tokens=None,
            finish_reason="stop",
            request_id=None,
            tool_calls=tool_calls,
        )
