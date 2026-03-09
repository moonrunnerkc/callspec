"""Google Generative AI provider adapter.

Wraps the google-generativeai Python SDK (>=0.3.0). Normalizes responses
into ProviderResponse. Google's API uses a different structure from the
OpenAI-style chat completions, so the adapter handles the translation
between Google's content parts and the standard ProviderResponse format.

The google-generativeai SDK uses generation_config for parameters like
temperature. The model string returned in the response is used as-is
for the ProviderResponse.model field.

Requires: pip install llm-assert[google]
"""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_assert.core.types import ProviderResponse
from llm_assert.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """Provider adapter for Google's Generative AI API.

    Accepts any model available through the google-generativeai SDK
    (gemini-2.0-flash, gemini-1.5-pro, etc.). The model object is
    initialized lazily on first call.
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_output_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> None:
        self._model_name = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._generation_kwargs = generation_kwargs
        self._model = None
        self._genai = None

    @property
    def provider_name(self) -> str:
        return "google"

    def _configure_sdk(self):
        """Configure the google-generativeai SDK and create the model."""
        if self._model is not None:
            return self._model

        try:
            import google.generativeai as genai
        except ImportError as import_error:
            raise ImportError(
                "Google provider requires the google-generativeai package. "
                "Install with: pip install llm-assert[google]"
            ) from import_error

        if self._api_key is not None:
            genai.configure(api_key=self._api_key)

        self._genai = genai

        generation_config = {"temperature": self._temperature}
        if self._max_output_tokens is not None:
            generation_config["max_output_tokens"] = self._max_output_tokens
        generation_config.update(self._generation_kwargs)

        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=generation_config,
        )
        return self._model

    def _build_contents(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> list:
        """Convert messages to Google's content format.

        Google uses a different role naming: 'model' instead of 'assistant',
        and system instructions are set at model level, not in content.
        """
        if messages is not None:
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                # Map OpenAI-style roles to Google roles
                if role == "assistant":
                    role = "model"
                elif role == "system":
                    # System messages get prepended to the first user message
                    # since Google handles system prompts via model config
                    continue
                contents.append({
                    "role": role,
                    "parts": [msg["content"]],
                })

            if not contents:
                contents = [{"role": "user", "parts": [prompt]}]

            return contents

        return [prompt]

    def call(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        # Google's generate_content does not accept seed or temperature as
        # call-time kwargs (they live in GenerationConfig at init time).
        kwargs.pop("seed", None)
        kwargs.pop("temperature", None)

        model = self._configure_sdk()
        contents = self._build_contents(prompt, messages)

        start = time.monotonic()
        response = model.generate_content(contents, **kwargs)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        # Extract text from the response
        content_text = response.text if response.text else ""

        # Build raw dict for inspection
        raw = {
            "model": self._model_name,
            "candidates": [],
        }
        if response.candidates:
            for candidate in response.candidates:
                raw["candidates"].append({
                    "content": str(candidate.content) if candidate.content else "",
                    "finish_reason": (
                        str(candidate.finish_reason)
                        if candidate.finish_reason else None
                    ),
                })

        # Token counts from usage metadata
        prompt_tokens = None
        completion_tokens = None
        if response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        finish_reason = None
        if response.candidates:
            raw_reason = response.candidates[0].finish_reason
            if raw_reason is not None:
                finish_reason = str(raw_reason)

        return ProviderResponse(
            content=content_text,
            raw=raw,
            model=self._model_name,
            provider="google",
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            request_id=None,
            tool_calls=self._extract_tool_calls(response),
        )

    async def call_async(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Async call using google-generativeai's async support.

        The google-generativeai SDK provides generate_content_async for
        async operations.
        """
        kwargs.pop("seed", None)
        kwargs.pop("temperature", None)

        model = self._configure_sdk()
        contents = self._build_contents(prompt, messages)

        start = time.monotonic()
        response = await model.generate_content_async(contents, **kwargs)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        content_text = response.text if response.text else ""

        raw = {
            "model": self._model_name,
            "candidates": [],
        }
        if response.candidates:
            for candidate in response.candidates:
                raw["candidates"].append({
                    "content": str(candidate.content) if candidate.content else "",
                    "finish_reason": (
                        str(candidate.finish_reason)
                        if candidate.finish_reason else None
                    ),
                })

        prompt_tokens = None
        completion_tokens = None
        if response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        finish_reason = None
        if response.candidates:
            raw_reason = response.candidates[0].finish_reason
            if raw_reason is not None:
                finish_reason = str(raw_reason)

        return ProviderResponse(
            content=content_text,
            raw=raw,
            model=self._model_name,
            provider="google",
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            request_id=None,
            tool_calls=self._extract_tool_calls(response),
        )

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        """Extract tool calls from a Google Generative AI response.

        Gemini returns function calls as Part objects with a function_call
        attribute containing name and args (a dict). The parts live inside
        response.candidates[0].content.parts.
        """
        extracted: list[dict[str, Any]] = []
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return extracted

        content = getattr(candidates[0], "content", None)
        if content is None:
            return extracted

        parts = getattr(content, "parts", None) or []
        for part in parts:
            fn_call = getattr(part, "function_call", None)
            if fn_call is None:
                continue
            name = getattr(fn_call, "name", "")
            # Gemini's function_call.args is a proto MapComposite; convert to dict
            args = getattr(fn_call, "args", {})
            if hasattr(args, "items"):
                args = dict(args)
            elif not isinstance(args, dict):
                args = {}
            extracted.append({
                "name": name,
                "arguments": args,
            })
        return extracted
