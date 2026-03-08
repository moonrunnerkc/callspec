"""MockProvider: deterministic function-based provider for testing.

No network calls. Takes a function that maps (prompt, messages) to a string
and wraps the output in a ProviderResponse with stable mock metadata.
Used in Verdict's own test suite and by teams testing assertion configurations
without spending API credits.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from verdict.core.types import ProviderResponse
from verdict.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Provider that returns deterministic responses from a user-supplied function.

    The response function receives (prompt, messages) and returns a string.
    That string becomes the `content` of a ProviderResponse with zeroed-out
    latency and token counts. The `model` field is "mock" and the `provider`
    field is "mock".
    """

    def __init__(
        self,
        response_fn: Callable[..., str],
        model_name: str = "mock",
        latency_ms: int = 0,
    ) -> None:
        self._response_fn = response_fn
        self._model_name = model_name
        self._latency_ms = latency_ms

    @property
    def provider_name(self) -> str:
        return "mock"

    def call(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        start = time.monotonic()
        content = self._response_fn(prompt, messages)
        elapsed_ms = int((time.monotonic() - start) * 1000) + self._latency_ms

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
        )
