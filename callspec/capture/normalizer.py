"""Tool-call normalizer: converts any supported source into a ToolCallTrajectory.

This is the central entry point for the capture layer. Given a response from
any supported source (ProviderResponse, raw OpenAI dict, raw Anthropic dict,
LangChain message, Pydantic AI response, or a plain list of tool-call dicts),
the normalizer produces a consistent ToolCallTrajectory.

The normalizer works on already-captured response data. It does not hook into
live API calls. Users pass in a response they already have and get back a
normalized trajectory. The interceptor (separate module) is a convenience
layer that wraps a provider call and feeds the result through here.
"""

from __future__ import annotations

from typing import Any

from callspec.core.trajectory import ToolCall, ToolCallTrajectory
from callspec.core.types import ProviderResponse


def normalize(
    source: Any,
    *,
    provider_hint: str | None = None,
) -> ToolCallTrajectory:
    """Convert a response source into a ToolCallTrajectory.

    Auto-detects the source format and dispatches to the appropriate adapter.
    When auto-detection is ambiguous, use provider_hint to force a format.

    Args:
        source: One of: ProviderResponse, raw dict, LangChain AIMessage,
            list of tool-call dicts, or any supported response object.
        provider_hint: Force interpretation as a specific format.
            Values: "openai", "anthropic", "langchain", "pydantic_ai", "generic".

    Returns:
        A ToolCallTrajectory with all tool calls extracted and indexed.

    Raises:
        ValueError: When the source format cannot be detected or is unsupported.
    """
    # ProviderResponse from our own provider adapters (already normalized)
    if isinstance(source, ProviderResponse):
        return _from_provider_response(source)

    # Plain list of tool-call dicts (generic format)
    if isinstance(source, list):
        return _from_generic_list(source)

    # Dict-based source: detect format from structure
    if isinstance(source, dict):
        return _from_dict(source, provider_hint)

    # Try LangChain AIMessage detection (duck-typed to avoid hard dependency)
    if _is_langchain_message(source):
        from callspec.capture.adapters.langchain import extract_from_message
        return extract_from_message(source)

    # Try Pydantic AI ModelResponse detection
    if _is_pydantic_ai_response(source):
        from callspec.capture.adapters.pydantic_ai import extract_from_response
        return extract_from_response(source)

    raise ValueError(
        f"Cannot normalize source of type {type(source).__name__}. "
        f"Supported types: ProviderResponse, dict, list[dict], "
        f"LangChain AIMessage, Pydantic AI ModelResponse."
    )


def _from_provider_response(response: ProviderResponse) -> ToolCallTrajectory:
    """Build trajectory from a ProviderResponse with pre-extracted tool_calls."""
    calls = []
    for index, tc in enumerate(response.tool_calls):
        calls.append(ToolCall(
            tool_name=tc.get("name", ""),
            arguments=tc.get("arguments", {}),
            call_index=index,
            model=response.model,
            provider=response.provider,
            call_id=tc.get("id"),
        ))

    return ToolCallTrajectory(
        calls=calls,
        model=response.model,
        provider=response.provider,
        raw_response=response.raw,
    )


def _from_generic_list(tool_call_dicts: list[Any]) -> ToolCallTrajectory:
    """Build trajectory from a plain list of tool-call dicts.

    Each dict should have at minimum {"name": str, "arguments": dict}.
    """
    calls = []
    for index, tc in enumerate(tool_call_dicts):
        if not isinstance(tc, dict):
            raise ValueError(
                f"Expected dict at position {index}, got {type(tc).__name__}. "
                f"Each tool call must be a dict with 'name' and 'arguments' keys."
            )
        calls.append(ToolCall(
            tool_name=tc.get("name", tc.get("tool_name")) or "",
            arguments=tc.get("arguments", tc.get("args")) or {},
            call_index=index,
            call_id=tc.get("id"),
        ))

    return ToolCallTrajectory(calls=calls)


def _from_dict(source: dict[str, Any], provider_hint: str | None) -> ToolCallTrajectory:
    """Detect and extract tool calls from a raw response dict.

    Detection logic:
    - Has "choices" key -> OpenAI Chat Completions format
    - Has "content" list + type=="message" or tool_use blocks -> Anthropic format
    - Has "output" key with list -> OpenAI Responses API format
    - Has "calls" key -> serialized ToolCallTrajectory
    """
    if provider_hint == "openai" or "choices" in source:
        from callspec.capture.adapters.openai import extract_from_dict
        return extract_from_dict(source)

    if provider_hint == "anthropic" or (
        "content" in source
        and isinstance(source.get("content"), list)
        and (
            # Anthropic responses have type=="message" at the top level
            source.get("type") == "message"
            or any(
                isinstance(block, dict) and block.get("type") == "tool_use"
                for block in source["content"]
            )
        )
    ):
        from callspec.capture.adapters.anthropic import extract_from_dict
        return extract_from_dict(source)

    # Responses API format: "output" is a list
    if "output" in source and isinstance(source.get("output"), list):
        from callspec.capture.adapters.openai import extract_from_dict
        return extract_from_dict(source)

    # Serialized ToolCallTrajectory
    if "calls" in source:
        return ToolCallTrajectory.from_dict(source)

    raise ValueError(
        "Cannot detect format of response dict. "
        "Expected OpenAI (has 'choices'), Anthropic (has 'content' with tool_use blocks), "
        "OpenAI Responses API (has 'output'), or serialized trajectory (has 'calls'). "
        f"Found top-level keys: {list(source.keys())}. "
        "Use provider_hint='openai' or provider_hint='anthropic' to force format."
    )


def _is_langchain_message(source: Any) -> bool:
    """Duck-type check for LangChain AIMessage without importing langchain."""
    return (
        hasattr(source, "tool_calls")
        and hasattr(source, "content")
        and hasattr(source, "type")
        and getattr(source, "type", None) == "ai"
    )


def _is_pydantic_ai_response(source: Any) -> bool:
    """Duck-type check for Pydantic AI ModelResponse without importing pydantic_ai."""
    return (
        hasattr(source, "parts")
        and hasattr(source, "model_name")
    )
