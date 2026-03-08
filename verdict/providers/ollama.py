"""Ollama provider adapter.

Wraps the ollama Python SDK (>=0.1.0) for local model inference.
Ollama runs models locally, so there are no API keys, no rate limits,
and no network calls to external services. Response times depend
entirely on local hardware.

Ollama does not return token counts in the standard response.
The prompt_tokens and completion_tokens fields will be populated
when the server returns eval_count/prompt_eval_count metadata,
which depends on the Ollama version and model.

Requires: pip install verdict[ollama]
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from verdict.core.types import ProviderResponse
from verdict.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Provider adapter for locally-hosted Ollama models.

    Connects to a running Ollama server (default: http://localhost:11434).
    Accepts any model pulled into the local Ollama instance (llama3,
    mistral, phi3, etc.).
    """

    def __init__(
        self,
        model: str = "llama3",
        host: Optional[str] = None,
        temperature: float = 0.0,
        seed: Optional[int] = 42,
        num_predict: Optional[int] = None,
        **client_kwargs: Any,
    ) -> None:
        self._model = model
        self._host = host
        self._temperature = temperature
        self._seed = seed
        self._num_predict = num_predict
        self._client_kwargs = client_kwargs
        self._client = None
        self._async_client = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from ollama import Client
        except ImportError as import_error:
            raise ImportError(
                "Ollama provider requires the ollama package. "
                "Install with: pip install verdict[ollama]"
            ) from import_error

        kwargs = {**self._client_kwargs}
        if self._host is not None:
            kwargs["host"] = self._host

        self._client = Client(**kwargs)
        return self._client

    def _get_async_client(self):
        if self._async_client is not None:
            return self._async_client

        try:
            from ollama import AsyncClient
        except ImportError as import_error:
            raise ImportError(
                "Ollama provider requires the ollama package. "
                "Install with: pip install verdict[ollama]"
            ) from import_error

        kwargs = {**self._client_kwargs}
        if self._host is not None:
            kwargs["host"] = self._host

        self._async_client = AsyncClient(**kwargs)
        return self._async_client

    def _build_messages(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        if messages is not None:
            return list(messages)
        return [{"role": "user", "content": prompt}]

    def _build_options(self, **kwargs) -> Dict[str, Any]:
        """Build Ollama options dict for temperature, seed, and num_predict."""
        options: Dict[str, Any] = {
            "temperature": self._temperature,
        }

        if self._seed is not None:
            options["seed"] = self._seed

        if self._num_predict is not None:
            options["num_predict"] = self._num_predict

        options.update(kwargs)
        return options

    def call(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        built_messages = self._build_messages(prompt, messages)
        options = self._build_options(**kwargs.pop("options", {}))

        start = time.monotonic()
        response = client.chat(
            model=self._model,
            messages=built_messages,
            options=options,
            **kwargs,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Ollama returns a dict-like response object
        raw_dict = dict(response) if not isinstance(response, dict) else response
        message_content = raw_dict.get("message", {})
        content = message_content.get("content", "") if isinstance(message_content, dict) else ""

        # Token counts when available from Ollama metadata
        prompt_tokens = raw_dict.get("prompt_eval_count")
        completion_tokens = raw_dict.get("eval_count")

        return ProviderResponse(
            content=content,
            raw=raw_dict,
            model=raw_dict.get("model", self._model),
            provider="ollama",
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=raw_dict.get("done_reason", "stop" if raw_dict.get("done") else None),
            request_id=None,
        )

    async def call_async(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Native async using Ollama's AsyncClient."""
        client = self._get_async_client()
        built_messages = self._build_messages(prompt, messages)
        options = self._build_options(**kwargs.pop("options", {}))

        start = time.monotonic()
        response = await client.chat(
            model=self._model,
            messages=built_messages,
            options=options,
            **kwargs,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        raw_dict = dict(response) if not isinstance(response, dict) else response
        message_content = raw_dict.get("message", {})
        content = message_content.get("content", "") if isinstance(message_content, dict) else ""

        prompt_tokens = raw_dict.get("prompt_eval_count")
        completion_tokens = raw_dict.get("eval_count")

        return ProviderResponse(
            content=content,
            raw=raw_dict,
            model=raw_dict.get("model", self._model),
            provider="ollama",
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=raw_dict.get("done_reason", "stop" if raw_dict.get("done") else None),
            request_id=None,
        )
