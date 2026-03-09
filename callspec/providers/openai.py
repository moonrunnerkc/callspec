"""OpenAI provider adapter.

Wraps the openai Python SDK (>=1.0.0). Normalizes responses into
ProviderResponse, extracting the actual model identifier returned by
the API (not the alias the developer requested) so model drift is
visible in assertion history.

Supports deterministic output via temperature=0 and the seed parameter.
OpenAI has supported deterministic seeds since November 2023.

Handles both Chat Completions and Responses API formats for tool call
extraction. Chat Completions returns tool calls in
choices[0].message.tool_calls. The Responses API (March 2025+) returns
them in response.output as items with type=="function_call".

Requires: pip install callspec[openai]
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from callspec.core.types import ProviderResponse
from callspec.providers.base import BaseProvider

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
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int | None = None,
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
        self._client: Any = None
        self._async_client: Any = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as import_error:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install callspec[openai]"
            ) from import_error

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            organization=self._organization,
            **self._client_kwargs,
        )
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is not None:
            return self._async_client

        try:
            from openai import AsyncOpenAI
        except ImportError as import_error:
            raise ImportError(
                "OpenAI provider requires the openai package. "
                "Install with: pip install callspec[openai]"
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
        messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Build the messages array from prompt and optional conversation history."""
        if messages is not None:
            return list(messages)
        return [{"role": "user", "content": prompt}]

    def _build_params(self, **kwargs: Any) -> dict[str, Any]:
        """Merge default params with per-call overrides."""
        params: dict[str, Any] = {
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

    @staticmethod
    def _extract_tool_calls(response_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract normalized tool calls from an OpenAI response dict.

        Handles two API formats:
        - Chat Completions: choices[0].message.tool_calls (objects with
          function.name and function.arguments)
        - Responses API (March 2025+): output items with type=="function_call"
          containing name, arguments (JSON string), and call_id
        """
        extracted: list[dict[str, Any]] = []

        # Chat Completions format
        choices = response_dict.get("choices", [])
        if choices:
            message = choices[0].get("message", {})

            # Current tool_calls format
            raw_tool_calls = message.get("tool_calls") or []
            for tc in raw_tool_calls:
                func = tc.get("function", {})
                args_str = func.get("arguments", "{}")
                try:
                    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": args_str}

                extracted.append({
                    "id": tc.get("id"),
                    "name": func.get("name", ""),
                    "arguments": arguments,
                })

            # Legacy function_call format (deprecated but still seen)
            if not extracted:
                function_call = message.get("function_call")
                if function_call:
                    args_str = function_call.get("arguments", "{}")
                    try:
                        arguments = (
                            json.loads(args_str) if isinstance(args_str, str) else args_str
                        )
                    except (json.JSONDecodeError, TypeError):
                        arguments = {"_raw": args_str}

                    extracted.append({
                        "name": function_call.get("name", ""),
                        "arguments": arguments,
                    })

            return extracted

        # Responses API format: output is a list of items
        output_items = response_dict.get("output", [])
        for item in output_items:
            if item.get("type") == "function_call":
                args_str = item.get("arguments", "{}")
                try:
                    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": args_str}

                extracted.append({
                    "id": item.get("call_id"),
                    "name": item.get("name", ""),
                    "arguments": arguments,
                })

        return extracted

    def call(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
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
        raw_dict = response.model_dump()

        return ProviderResponse(
            content=choice.message.content or "",
            raw=raw_dict,
            model=response.model,
            provider="openai",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
            request_id=response.id,
            tool_calls=self._extract_tool_calls(raw_dict),
        )

    async def call_async(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
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
        raw_dict = response.model_dump()

        return ProviderResponse(
            content=choice.message.content or "",
            raw=raw_dict,
            model=response.model,
            provider="openai",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
            request_id=response.id,
            tool_calls=self._extract_tool_calls(raw_dict),
        )
