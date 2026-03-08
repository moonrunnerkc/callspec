"""Mistral provider adapter.

Wraps the mistralai Python SDK (>=0.1.0). Normalizes responses into
ProviderResponse using the standard chat completions interface that
Mistral's API provides (structurally similar to OpenAI's).

Mistral supports temperature control but does not expose a seed
parameter for deterministic output. At temperature=0, outputs are
near-deterministic.

Requires: pip install verdict[mistral]
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from verdict.core.types import ProviderResponse
from verdict.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class MistralProvider(BaseProvider):
    """Provider adapter for Mistral's chat completions API.

    Accepts any model available through the Mistral API (mistral-large-latest,
    mistral-small-latest, open-mistral-nemo, etc.). The client is initialized
    lazily on first call.
    """

    def __init__(
        self,
        model: str = "mistral-large-latest",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **client_kwargs: Any,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client_kwargs = client_kwargs
        self._client = None

    @property
    def provider_name(self) -> str:
        return "mistral"

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from mistralai import Mistral
        except ImportError as import_error:
            raise ImportError(
                "Mistral provider requires the mistralai package. "
                "Install with: pip install verdict[mistral]"
            ) from import_error

        kwargs = {**self._client_kwargs}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key

        self._client = Mistral(**kwargs)
        return self._client

    def _build_messages(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        if messages is not None:
            return list(messages)
        return [{"role": "user", "content": prompt}]

    def _build_params(self, **kwargs) -> Dict[str, Any]:
        # Mistral does not support a seed parameter for deterministic output.
        kwargs.pop("seed", None)

        params: Dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
        }

        if self._max_tokens is not None:
            params["max_tokens"] = self._max_tokens

        params.update(kwargs)
        return params

    def call(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        built_messages = self._build_messages(prompt, messages)
        params = self._build_params(**kwargs)

        start = time.monotonic()
        response = client.chat.complete(
            messages=built_messages,
            **params,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        usage = response.usage

        return ProviderResponse(
            content=choice.message.content or "",
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            model=response.model or self._model,
            provider="mistral",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None,
            request_id=response.id if hasattr(response, "id") else None,
        )

    async def call_async(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Async call using Mistral's async chat complete."""
        client = self._get_client()
        built_messages = self._build_messages(prompt, messages)
        params = self._build_params(**kwargs)

        start = time.monotonic()
        response = await client.chat.complete_async(
            messages=built_messages,
            **params,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        usage = response.usage

        return ProviderResponse(
            content=choice.message.content or "",
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
            model=response.model or self._model,
            provider="mistral",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None,
            request_id=response.id if hasattr(response, "id") else None,
        )
