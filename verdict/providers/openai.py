"""OpenAI provider adapter.

Wraps the openai Python SDK (>=1.0.0). Normalizes responses into
ProviderResponse, extracting the actual model identifier returned by
the API (not the alias the developer requested) so model drift is
visible in assertion history.

Supports deterministic output via temperature=0 and the seed parameter.
OpenAI has supported deterministic seeds since November 2023.

Requires: pip install verdict[openai]
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from verdict.core.types import ProviderResponse
from verdict.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Provider adapter for OpenAI's chat completions API.

    Accepts any model available through the OpenAI API (gpt-4o, gpt-4o-mini,
    o1, o3-mini, etc.). The client is initialized lazily on first call to
    avoid import-time side effects when the provider is instantiated but
    not yet used.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
        max_tokens: Optional[int] = None,
        **client_kwargs: Any,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        self._client_kwargs = client_kwargs
        self._client = None
        self._async_client = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as import_error:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install verdict[openai]"
            ) from import_error

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            organization=self._organization,
            **self._client_kwargs,
        )
        return self._client

    def _get_async_client(self):
        if self._async_client is not None:
            return self._async_client

        try:
            from openai import AsyncOpenAI
        except ImportError as import_error:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install verdict[openai]"
            ) from import_error

        self._async_client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            organization=self._organization,
            **self._client_kwargs,
        )
        return self._async_client

    def _build_messages(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build the messages array from prompt and optional conversation history."""
        if messages is not None:
            return list(messages)
        return [{"role": "user", "content": prompt}]

    def _build_params(self, **kwargs: Any) -> Dict[str, Any]:
        """Merge default params with per-call overrides."""
        params: Dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
        }

        if self._seed is not None:
            params["seed"] = self._seed

        if self._max_tokens is not None:
            params["max_tokens"] = self._max_tokens

        # Per-call overrides take precedence
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
        response = client.chat.completions.create(
            messages=built_messages,
            **params,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        usage = response.usage

        return ProviderResponse(
            content=choice.message.content or "",
            raw=response.model_dump(),
            model=response.model,
            provider="openai",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
            request_id=response.id,
        )

    async def call_async(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Native async using OpenAI's AsyncOpenAI client."""
        client = self._get_async_client()
        built_messages = self._build_messages(prompt, messages)
        params = self._build_params(**kwargs)

        start = time.monotonic()
        response = await client.chat.completions.create(
            messages=built_messages,
            **params,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        usage = response.usage

        return ProviderResponse(
            content=choice.message.content or "",
            raw=response.model_dump(),
            model=response.model,
            provider="openai",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
            request_id=response.id,
        )
