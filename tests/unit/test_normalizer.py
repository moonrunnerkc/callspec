"""Tests for the capture normalizer: auto-detection and dispatch.

The normalizer is the central entry point. It must correctly detect the
source type (ProviderResponse, raw dict, list, LangChain, Pydantic AI)
and dispatch to the right adapter. These tests verify dispatch logic
and error handling, not individual adapter correctness (those are in
test_capture_adapters.py).
"""

from __future__ import annotations

from typing import Any

import pytest

from callspec.capture.normalizer import normalize
from callspec.core.trajectory import ToolCallTrajectory
from callspec.core.types import ProviderResponse
from tests.fixtures.tool_call_responses import (
    ANTHROPIC_NO_TOOL_USE,
    ANTHROPIC_SINGLE_TOOL_USE,
    GENERIC_TOOL_CALLS,
    GENERIC_TOOL_CALLS_ALTERNATE_KEYS,
    OPENAI_CHAT_NO_TOOL_CALLS,
    OPENAI_CHAT_PARALLEL_TOOL_CALLS,
    OPENAI_CHAT_SINGLE_TOOL_CALL,
    OPENAI_RESPONSES_API_SINGLE,
)

# ---------------------------------------------------------------------------
# ProviderResponse source
# ---------------------------------------------------------------------------

class TestNormalizeProviderResponse:
    """normalize() with a ProviderResponse containing pre-extracted tool_calls."""

    def test_with_tool_calls(self):
        response = ProviderResponse(
            content="",
            raw={"some": "data"},
            model="gpt-4o",
            provider="openai",
            tool_calls=[
                {"name": "search", "arguments": {"q": "flights"}, "id": "tc_1"},
                {"name": "book", "arguments": {"id": "BA123"}, "id": "tc_2"},
            ],
        )
        traj = normalize(response)
        assert isinstance(traj, ToolCallTrajectory)
        assert len(traj) == 2
        assert traj.tool_names == ["search", "book"]
        assert traj.model == "gpt-4o"
        assert traj.provider == "openai"
        assert traj.calls[0].call_id == "tc_1"

    def test_without_tool_calls(self):
        response = ProviderResponse(
            content="Just text",
            model="gpt-4o",
            provider="openai",
        )
        traj = normalize(response)
        assert traj.is_empty
        assert traj.model == "gpt-4o"

    def test_preserves_raw_response(self):
        raw = {"original": "full_response"}
        response = ProviderResponse(content="", raw=raw, tool_calls=[])
        traj = normalize(response)
        assert traj.raw_response is raw


# ---------------------------------------------------------------------------
# Dict source: auto-detection
# ---------------------------------------------------------------------------

class TestNormalizeDictAutoDetect:
    """normalize() with raw dicts, verifying format auto-detection."""

    def test_openai_chat_detected(self):
        traj = normalize(OPENAI_CHAT_SINGLE_TOOL_CALL)
        assert len(traj) == 1
        assert traj.tool_names == ["search_flights"]
        assert traj.provider == "openai"

    def test_openai_parallel_detected(self):
        traj = normalize(OPENAI_CHAT_PARALLEL_TOOL_CALLS)
        assert len(traj) == 3

    def test_openai_no_calls_detected(self):
        traj = normalize(OPENAI_CHAT_NO_TOOL_CALLS)
        assert traj.is_empty

    def test_openai_responses_api_detected(self):
        traj = normalize(OPENAI_RESPONSES_API_SINGLE)
        assert len(traj) == 1
        assert traj.tool_names == ["search_flights"]

    def test_anthropic_detected(self):
        traj = normalize(ANTHROPIC_SINGLE_TOOL_USE)
        assert len(traj) == 1
        assert traj.tool_names == ["search_flights"]
        assert traj.provider == "anthropic"

    def test_anthropic_no_calls_detected(self):
        traj = normalize(ANTHROPIC_NO_TOOL_USE)
        assert traj.is_empty

    def test_serialized_trajectory_detected(self):
        """A dict with 'calls' key is treated as a serialized ToolCallTrajectory."""
        data = {
            "calls": [
                {"tool_name": "search", "arguments": {"q": "test"}, "call_index": 0},
            ],
            "model": "gpt-4o",
            "provider": "openai",
        }
        traj = normalize(data)
        assert len(traj) == 1
        assert traj.tool_names == ["search"]
        assert traj.model == "gpt-4o"

    def test_unrecognized_dict_raises(self):
        with pytest.raises(ValueError, match="Cannot detect format"):
            normalize({"unknown_key": "value"})


# ---------------------------------------------------------------------------
# Dict source: provider_hint override
# ---------------------------------------------------------------------------

class TestNormalizeDictWithHint:
    """provider_hint forces interpretation regardless of auto-detection."""

    def test_openai_hint(self):
        traj = normalize(OPENAI_CHAT_SINGLE_TOOL_CALL, provider_hint="openai")
        assert traj.tool_names == ["search_flights"]

    def test_anthropic_hint(self):
        traj = normalize(ANTHROPIC_SINGLE_TOOL_USE, provider_hint="anthropic")
        assert traj.tool_names == ["search_flights"]


# ---------------------------------------------------------------------------
# List source
# ---------------------------------------------------------------------------

class TestNormalizeList:
    """normalize() with a plain list of tool-call dicts."""

    def test_standard_keys(self):
        traj = normalize(GENERIC_TOOL_CALLS)
        assert len(traj) == 2
        assert traj.tool_names == ["search", "book"]

    def test_alternate_keys(self):
        traj = normalize(GENERIC_TOOL_CALLS_ALTERNATE_KEYS)
        assert len(traj) == 2
        assert traj.tool_names == ["search", "book"]

    def test_empty_list(self):
        traj = normalize([])
        assert traj.is_empty

    def test_non_dict_in_list_raises(self):
        with pytest.raises(ValueError, match="Expected dict at position 0"):
            normalize(["not_a_dict"])


# ---------------------------------------------------------------------------
# Duck-typed LangChain source
# ---------------------------------------------------------------------------

class _FakeAIMessage:
    """Minimal duck-typed AIMessage for normalizer dispatch testing."""

    def __init__(self, tool_calls: list | None = None) -> None:
        self.content = "text"
        self.tool_calls = tool_calls or []
        self.type = "ai"
        self.response_metadata: dict[str, Any] = {}


class TestNormalizeLangChain:
    """normalize() dispatches to LangChain adapter for AIMessage-like objects."""

    def test_detected_and_extracted(self):
        msg = _FakeAIMessage(tool_calls=[
            {"name": "search", "args": {"q": "flights"}, "id": "lc_1"},
        ])
        traj = normalize(msg)
        assert len(traj) == 1
        assert traj.tool_names == ["search"]
        assert traj.calls[0].provider == "langchain"

    def test_empty_tool_calls(self):
        msg = _FakeAIMessage()
        traj = normalize(msg)
        assert traj.is_empty


# ---------------------------------------------------------------------------
# Duck-typed Pydantic AI source
# ---------------------------------------------------------------------------

class _FakeToolCallPart:
    def __init__(self, tool_name: str, args: dict, tool_call_id: str | None = None):
        self.tool_name = tool_name
        self.args = args
        self.tool_call_id = tool_call_id


class _FakeModelResponse:
    def __init__(self, parts: list, model_name: str = ""):
        self.parts = parts
        self.model_name = model_name


class TestNormalizePydanticAI:
    """normalize() dispatches to Pydantic AI adapter for ModelResponse-like objects."""

    def test_detected_and_extracted(self):
        resp = _FakeModelResponse(
            parts=[_FakeToolCallPart("search", {"q": "flights"}, "pa_1")],
            model_name="gpt-4o",
        )
        traj = normalize(resp)
        assert len(traj) == 1
        assert traj.tool_names == ["search"]
        assert traj.calls[0].provider == "pydantic_ai"

    def test_no_tool_parts(self):
        resp = _FakeModelResponse(parts=[], model_name="test")
        traj = normalize(resp)
        assert traj.is_empty


# ---------------------------------------------------------------------------
# Unsupported source
# ---------------------------------------------------------------------------

class TestNormalizeUnsupported:
    """normalize() raises clear errors for unsupported types."""

    def test_string_raises(self):
        with pytest.raises(ValueError, match="Cannot normalize source of type str"):
            normalize("not a response")

    def test_int_raises(self):
        with pytest.raises(ValueError, match="Cannot normalize source of type int"):
            normalize(42)

    def test_none_raises(self):
        with pytest.raises(ValueError, match="Cannot normalize source of type NoneType"):
            normalize(None)
