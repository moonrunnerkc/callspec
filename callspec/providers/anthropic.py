"""Anthropic provider adapter.

Wraps the anthropic Python SDK (>=0.18.0). Normalizes responses into
ProviderResponse, extracting the actual model identifier returned by
the API so model drift is visible in assertion history.

Important: Anthropic does not offer a seed parameter. At temperature=0,
Claude outputs are highly consistent but not perfectly deterministic.

Tool calls are extracted from content blocks where type=="tool_use".
Each tool_use block contains an id, name, and input dict that maps
directly to the normalized tool_calls format.

Requires: pip install callspec[anthropic]
"""

from __future__ import annotations

import logging
import time
from typing import Any

from callspec.core.types import ProviderResponse
from callspec.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Provider adapter for Anthropic's messages API.

    Accepts any model available through the Anthropic API (claude-sonnet-4-20250514,
    claude-3-5-haiku-20241022, etc.). The client is initialized lazily on
    first call to avoid import-time side effects.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        **client_kwargs: Any,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        # Anthropic requires max_tokens on every request
        self._max_tokens = max_tokens
        self._client_kwargs = client_kwargs
        self._client: Any = None
        self._async_client: Any = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from anthropic import Anthropic
        except ImportError as import_error:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install callspec[anthropic]"
            ) from import_error

        kwargs = {**self._client_kwargs}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key

        self._client = Anthropic(**kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is not None:
            return self._async_client

        try:
            from anthropic import AsyncAnthropic
        except ImportError as import_error:
            raise ImportError(
                "Anthropic provider requires the anthropic package. "
                "Install with: pip install callspec[anthropic]"
            ) from import_error

        kwargs = {**self._client_kwargs}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key

        self._async_client = AsyncAnthropic(**kwargs)
        return self._async_client

    def _extract_system_and_messages(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Split system message from conversation messages.

        Anthropic's API takes system as a separate top-level parameter,
        not as a message in the array. If the messages list contains a
        system role, extract it. Otherwise, use prompt as the user turn.
        """
        if messages is not None:
            system_text = None
            user_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_text = msg["content"]
                else:
                    user_messages.append(msg)

            if not user_messages:
                user_messages = [{"role": "user", "content": prompt}]

            return system_text, user_messages

        return None, [{"role": "user", "content": prompt}]

    def _build_params(
        self,
        system_text: str | None,
        user_messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request parameters for Anthropic's messages API."""
        # Anthropic does not support a seed parameter for deterministic output.
        # The runner passes seed= on every call; strip it here to avoid a
        # TypeError from the Anthropic SDK.
        kwargs.pop("seed", None)

        params: dict[str, Any] = {
            "model": self._model,
            "messages": user_messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }

        if system_text is not None:
            params["system"] = system_text

        params.update(kwargs)
        return params

    def call(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        system_text, user_messages = self._extract_system_and_messages(prompt, messages)
        params = self._build_params(system_text, user_messages, **kwargs)

        start = time.monotonic()
        response = client.messages.create(**params)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Anthropic returns content as a list of content blocks
        content_text = ""
        tool_calls_extracted: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls_extracted.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return ProviderResponse(
            content=content_text,
            raw=response.model_dump(),
            model=response.model,
            provider="anthropic",
            latency_ms=elapsed_ms,
            prompt_tokens=response.usage.input_tokens if response.usage else None,
            completion_tokens=response.usage.output_tokens if response.usage else None,
            finish_reason=response.stop_reason,
            request_id=response.id,
            tool_calls=tool_calls_extracted,
        )

    async def call_async(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Native async using Anthropic's AsyncAnthropic client."""
        client = self._get_async_client()
        system_text, user_messages = self._extract_system_and_messages(prompt, messages)
        params = self._build_params(system_text, user_messages, **kwargs)

        start = time.monotonic()
        response = await client.messages.create(**params)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        content_text = ""
        tool_calls_extracted: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                tool_calls_extracted.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return ProviderResponse(
            content=content_text,
            raw=response.model_dump(),
            model=response.model,
            provider="anthropic",
            latency_ms=elapsed_ms,
            prompt_tokens=response.usage.input_tokens if response.usage else None,
            completion_tokens=response.usage.output_tokens if response.usage else None,
            finish_reason=response.stop_reason,
            request_id=response.id,
            tool_calls=tool_calls_extracted,
        )
