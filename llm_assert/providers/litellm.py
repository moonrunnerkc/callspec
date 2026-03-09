"""LiteLLM provider adapter (catch-all).

Wraps the litellm Python SDK (>=1.0.0) which provides a unified interface
to 100+ LLM providers through a single API. This is the catch-all adapter
for any provider not covered by LLMAssert's first-party adapters.

LiteLLM normalizes provider APIs into an OpenAI-compatible interface,
so the adapter code closely mirrors the OpenAI adapter. The key advantage
is that developers can use any LiteLLM-supported provider (Cohere,
AI21, Replicate, Together, Perplexity, etc.) without waiting for a
dedicated LLMAssert adapter.

The model string follows LiteLLM's provider/model naming convention:
"anthropic/claude-sonnet-4-20250514", "ollama/llama3", "together_ai/meta-llama/Llama-3-70b",
etc. See https://docs.litellm.ai/docs/providers for the full list.

Requires: pip install llm-assert[litellm]
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from llm_assert.core.types import ProviderResponse
from llm_assert.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class LiteLLMProvider(BaseProvider):
    """Catch-all provider adapter using LiteLLM's unified API.

    Routes requests to any LiteLLM-supported provider. The model string
    determines the provider: "gpt-4o" routes to OpenAI, "anthropic/claude-3-opus"
    routes to Anthropic, "ollama/llama3" routes to Ollama, etc.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        api_base: str | None = None,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int | None = None,
        **litellm_kwargs: Any,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._temperature = temperature
        self._seed = seed
        self._max_tokens = max_tokens
        self._litellm_kwargs = litellm_kwargs
        self._litellm = None

    @property
    def provider_name(self) -> str:
        return "litellm"

    def _get_litellm(self):
        if self._litellm is not None:
            return self._litellm

        try:
            import litellm
        except ImportError as import_error:
            raise ImportError(
                "LiteLLM provider requires the litellm package. "
                "Install with: pip install llm-assert[litellm]"
            ) from import_error

        self._litellm = litellm
        return litellm

    def _build_messages(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        if messages is not None:
            return list(messages)
        return [{"role": "user", "content": prompt}]

    def _build_params(self, **kwargs) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
        }

        if self._seed is not None:
            params["seed"] = self._seed

        if self._max_tokens is not None:
            params["max_tokens"] = self._max_tokens

        if self._api_key is not None:
            params["api_key"] = self._api_key

        if self._api_base is not None:
            params["api_base"] = self._api_base

        # LiteLLM-specific kwargs (custom_llm_provider, etc.)
        params.update(self._litellm_kwargs)

        # Per-call overrides
        params.update(kwargs)
        return params

    def _extract_provider_from_model(self) -> str:
        """Derive the underlying provider name from the LiteLLM model string.

        Used for the raw metadata, not for routing (LiteLLM handles routing).
        """
        if "/" in self._model:
            return self._model.split("/")[0]
        return "litellm"

    def call(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        litellm = self._get_litellm()
        built_messages = self._build_messages(prompt, messages)
        params = self._build_params(**kwargs)

        start = time.monotonic()
        response = litellm.completion(
            messages=built_messages,
            **params,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        usage = response.usage

        # LiteLLM returns the actual model from the provider when available
        actual_model = getattr(response, "model", self._model) or self._model

        return ProviderResponse(
            content=choice.message.content or "",
            raw=response.model_dump() if hasattr(response, "model_dump") else dict(response),
            model=actual_model,
            provider="litellm",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None,
            request_id=getattr(response, "id", None),
            tool_calls=self._extract_tool_calls(choice),
        )

    async def call_async(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Async call using LiteLLM's acompletion."""
        litellm = self._get_litellm()
        built_messages = self._build_messages(prompt, messages)
        params = self._build_params(**kwargs)

        start = time.monotonic()
        response = await litellm.acompletion(
            messages=built_messages,
            **params,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        choice = response.choices[0]
        usage = response.usage

        actual_model = getattr(response, "model", self._model) or self._model

        return ProviderResponse(
            content=choice.message.content or "",
            raw=response.model_dump() if hasattr(response, "model_dump") else dict(response),
            model=actual_model,
            provider="litellm",
            latency_ms=elapsed_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason if hasattr(choice, "finish_reason") else None,
            request_id=getattr(response, "id", None),
            tool_calls=self._extract_tool_calls(choice),
        )

    @staticmethod
    def _extract_tool_calls(choice: Any) -> list[dict[str, Any]]:
        """Extract tool calls from a LiteLLM choice object.

        LiteLLM normalizes all providers into an OpenAI-compatible format,
        so tool_calls appear on choice.message.tool_calls with the standard
        {id, type, function: {name, arguments}} structure.
        """
        raw_calls = getattr(getattr(choice, "message", None), "tool_calls", None) or []
        extracted: list[dict[str, Any]] = []
        for tc in raw_calls:
            func = getattr(tc, "function", None)
            if func is None:
                func = tc.get("function", {}) if isinstance(tc, dict) else {}
            func_name = getattr(func, "name", "") if not isinstance(func, dict) else func.get("name", "")
            arguments = getattr(func, "arguments", "{}") if not isinstance(func, dict) else func.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    arguments = {"_raw": arguments}
            tc_id = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
            extracted.append({
                "name": func_name,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "id": tc_id,
            })
        return extracted
