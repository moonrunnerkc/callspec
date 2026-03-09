"""Tests for all capture adapters: OpenAI, Anthropic, LangChain, Pydantic AI, generic.

Each adapter is tested against recorded response fixtures that mirror
real provider response structures. Edge cases (no tool calls, malformed
arguments, legacy formats) are all covered.
"""

from __future__ import annotations

from callspec.capture.adapters.anthropic import extract_from_dict as anthropic_extract
from callspec.capture.adapters.generic import extract_from_list as generic_extract
from callspec.capture.adapters.langchain import extract_from_message as langchain_extract
from callspec.capture.adapters.openai import extract_from_dict as openai_extract
from callspec.capture.adapters.pydantic_ai import extract_from_response as pydantic_ai_extract
from tests.fixtures.tool_call_responses import (
    ANTHROPIC_MULTIPLE_TOOL_USE,
    ANTHROPIC_NO_TOOL_USE,
    ANTHROPIC_SINGLE_TOOL_USE,
    GENERIC_TOOL_CALLS,
    GENERIC_TOOL_CALLS_ALTERNATE_KEYS,
    OPENAI_CHAT_LEGACY_FUNCTION_CALL,
    OPENAI_CHAT_MALFORMED_ARGS,
    OPENAI_CHAT_NO_TOOL_CALLS,
    OPENAI_CHAT_PARALLEL_TOOL_CALLS,
    OPENAI_CHAT_SINGLE_TOOL_CALL,
    OPENAI_RESPONSES_API_MULTIPLE,
    OPENAI_RESPONSES_API_SINGLE,
)

# ---------------------------------------------------------------------------
# OpenAI Chat Completions adapter
# ---------------------------------------------------------------------------

class TestOpenAIAdapter:
    """OpenAI adapter across Chat Completions, legacy, and Responses API."""

    def test_single_tool_call(self):
        traj = openai_extract(OPENAI_CHAT_SINGLE_TOOL_CALL)
        assert len(traj) == 1
        assert traj.tool_names == ["search_flights"]
        assert traj.provider == "openai"
        assert traj.model == "gpt-4o-2024-11-20"

        call = traj.calls[0]
        assert call.arguments == {
            "origin": "NYC",
            "destination": "London",
            "date": "2026-03-15",
        }
        assert call.call_id == "call_xyz789"
        assert call.call_index == 0

    def test_parallel_tool_calls(self):
        traj = openai_extract(OPENAI_CHAT_PARALLEL_TOOL_CALLS)
        assert len(traj) == 3
        assert traj.tool_names == ["get_weather", "get_weather", "convert_currency"]
        assert traj.call_count("get_weather") == 2
        assert traj.call_count("convert_currency") == 1

        # Verify sequential indexing
        for expected_index, call in enumerate(traj.calls):
            assert call.call_index == expected_index

        # Verify arguments on the currency call
        currency_call = traj.calls_to("convert_currency")[0]
        assert currency_call.arguments == {"from": "USD", "to": "EUR", "amount": 100}

    def test_legacy_function_call(self):
        traj = openai_extract(OPENAI_CHAT_LEGACY_FUNCTION_CALL)
        assert len(traj) == 1
        assert traj.tool_names == ["get_user_info"]
        assert traj.calls[0].arguments == {"user_id": "u_12345"}
        assert traj.model == "gpt-3.5-turbo-0613"

    def test_no_tool_calls_returns_empty(self):
        traj = openai_extract(OPENAI_CHAT_NO_TOOL_CALLS)
        assert traj.is_empty
        assert traj.model == "gpt-4o-2024-11-20"
        assert traj.provider == "openai"

    def test_responses_api_single(self):
        traj = openai_extract(OPENAI_RESPONSES_API_SINGLE)
        assert len(traj) == 1
        assert traj.tool_names == ["search_flights"]
        assert traj.calls[0].call_id == "fc_xyz789"
        assert traj.calls[0].arguments == {
            "origin": "NYC",
            "destination": "London",
        }
        assert traj.model == "gpt-4o-2025-03-01"

    def test_responses_api_multiple_with_text(self):
        """Non-function_call output items (message blocks) are skipped."""
        traj = openai_extract(OPENAI_RESPONSES_API_MULTIPLE)
        assert len(traj) == 2
        assert traj.tool_names == ["search_flights", "check_availability"]
        assert traj.calls[0].call_index == 0
        assert traj.calls[1].call_index == 1

    def test_malformed_arguments_preserved_as_raw(self):
        """Unparseable JSON arguments are stored under _raw instead of raising."""
        traj = openai_extract(OPENAI_CHAT_MALFORMED_ARGS)
        assert len(traj) == 1
        assert traj.calls[0].tool_name == "broken_tool"
        assert "_raw" in traj.calls[0].arguments
        assert traj.calls[0].arguments["_raw"] == "not valid json {{{"

    def test_raw_response_preserved(self):
        traj = openai_extract(OPENAI_CHAT_SINGLE_TOOL_CALL)
        assert traj.raw_response is OPENAI_CHAT_SINGLE_TOOL_CALL

    def test_empty_dict_returns_empty(self):
        traj = openai_extract({})
        assert traj.is_empty


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class TestAnthropicAdapter:
    """Anthropic adapter for tool_use content blocks."""

    def test_single_tool_use(self):
        traj = anthropic_extract(ANTHROPIC_SINGLE_TOOL_USE)
        assert len(traj) == 1
        assert traj.tool_names == ["search_flights"]
        assert traj.provider == "anthropic"
        assert traj.model == "claude-sonnet-4-20250514"

        call = traj.calls[0]
        assert call.arguments == {
            "origin": "NYC",
            "destination": "London",
            "date": "2026-03-15",
        }
        assert call.call_id == "toolu_xyz789"

    def test_multiple_tool_use_with_interleaved_text(self):
        traj = anthropic_extract(ANTHROPIC_MULTIPLE_TOOL_USE)
        assert len(traj) == 2
        assert traj.tool_names == ["search_flights", "search_hotels"]
        assert traj.calls[0].call_index == 0
        assert traj.calls[1].call_index == 1
        assert traj.calls[1].arguments == {
            "city": "London",
            "checkin": "2026-03-15",
        }

    def test_no_tool_use_returns_empty(self):
        traj = anthropic_extract(ANTHROPIC_NO_TOOL_USE)
        assert traj.is_empty
        assert traj.model == "claude-sonnet-4-20250514"
        assert traj.provider == "anthropic"

    def test_raw_response_preserved(self):
        traj = anthropic_extract(ANTHROPIC_SINGLE_TOOL_USE)
        assert traj.raw_response is ANTHROPIC_SINGLE_TOOL_USE

    def test_empty_content_list(self):
        data = {"model": "claude", "content": []}
        traj = anthropic_extract(data)
        assert traj.is_empty


# ---------------------------------------------------------------------------
# LangChain adapter (duck-typed)
# ---------------------------------------------------------------------------

class _FakeAIMessage:
    """Minimal duck-typed stand-in for LangChain's AIMessage."""

    def __init__(
        self,
        content: str = "",
        tool_calls: list | None = None,
        response_metadata: dict | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.type = "ai"
        self.response_metadata = response_metadata or {}


class TestLangChainAdapter:
    """LangChain adapter using duck-typed AIMessage stand-ins."""

    def test_single_tool_call(self):
        msg = _FakeAIMessage(
            tool_calls=[
                {"name": "search", "args": {"q": "flights"}, "id": "lc_001"},
            ],
            response_metadata={"model_name": "gpt-4o"},
        )
        traj = langchain_extract(msg)
        assert len(traj) == 1
        assert traj.tool_names == ["search"]
        assert traj.calls[0].arguments == {"q": "flights"}
        assert traj.calls[0].call_id == "lc_001"
        assert traj.calls[0].provider == "langchain"
        assert traj.model == "gpt-4o"

    def test_multiple_tool_calls(self):
        msg = _FakeAIMessage(
            tool_calls=[
                {"name": "search", "args": {"q": "a"}, "id": "lc_001"},
                {"name": "book", "args": {"id": "B1"}, "id": "lc_002"},
                {"name": "confirm", "args": {}, "id": "lc_003"},
            ],
        )
        traj = langchain_extract(msg)
        assert len(traj) == 3
        assert traj.tool_names == ["search", "book", "confirm"]
        for expected_index, call in enumerate(traj.calls):
            assert call.call_index == expected_index

    def test_no_tool_calls(self):
        msg = _FakeAIMessage(content="Just text, no tools.")
        traj = langchain_extract(msg)
        assert traj.is_empty

    def test_missing_model_uses_empty_string(self):
        msg = _FakeAIMessage(
            tool_calls=[{"name": "x", "args": {}}],
        )
        traj = langchain_extract(msg)
        assert traj.model == ""


# ---------------------------------------------------------------------------
# Pydantic AI adapter (duck-typed)
# ---------------------------------------------------------------------------

class _FakeToolCallPart:
    """Stand-in for Pydantic AI's ToolCallPart."""

    def __init__(self, tool_name: str, args: dict, tool_call_id: str | None = None):
        self.tool_name = tool_name
        self.args = args
        self.tool_call_id = tool_call_id


class _FakeTextPart:
    """Stand-in for a non-tool part in ModelResponse."""

    def __init__(self, text: str):
        self.text = text


class _FakeModelResponse:
    """Stand-in for Pydantic AI's ModelResponse."""

    def __init__(self, parts: list, model_name: str = ""):
        self.parts = parts
        self.model_name = model_name


class TestPydanticAIAdapter:
    """Pydantic AI adapter using duck-typed stand-ins."""

    def test_single_tool_call(self):
        resp = _FakeModelResponse(
            parts=[
                _FakeToolCallPart("search", {"q": "flights"}, "pa_001"),
            ],
            model_name="gpt-4o",
        )
        traj = pydantic_ai_extract(resp)
        assert len(traj) == 1
        assert traj.tool_names == ["search"]
        assert traj.calls[0].arguments == {"q": "flights"}
        assert traj.calls[0].call_id == "pa_001"
        assert traj.calls[0].provider == "pydantic_ai"

    def test_mixed_parts_skips_non_tool(self):
        resp = _FakeModelResponse(
            parts=[
                _FakeTextPart("I'll search for you."),
                _FakeToolCallPart("search", {"q": "flights"}),
                _FakeTextPart("Found results."),
                _FakeToolCallPart("book", {"id": "BA123"}),
            ],
            model_name="claude",
        )
        traj = pydantic_ai_extract(resp)
        assert len(traj) == 2
        assert traj.tool_names == ["search", "book"]
        assert traj.calls[0].call_index == 0
        assert traj.calls[1].call_index == 1

    def test_no_tool_parts(self):
        resp = _FakeModelResponse(parts=[_FakeTextPart("Just text.")])
        traj = pydantic_ai_extract(resp)
        assert traj.is_empty

    def test_empty_parts(self):
        resp = _FakeModelResponse(parts=[], model_name="test")
        traj = pydantic_ai_extract(resp)
        assert traj.is_empty
        assert traj.model == "test"


# ---------------------------------------------------------------------------
# Generic adapter
# ---------------------------------------------------------------------------

class TestGenericAdapter:
    """Generic adapter for plain lists of tool-call dicts."""

    def test_standard_keys(self):
        traj = generic_extract(GENERIC_TOOL_CALLS)
        assert len(traj) == 2
        assert traj.tool_names == ["search", "book"]
        assert traj.calls[0].arguments == {"query": "flights to London"}
        assert traj.calls[1].arguments == {"flight_id": "BA123", "passenger": "Brad"}

    def test_alternate_keys(self):
        traj = generic_extract(GENERIC_TOOL_CALLS_ALTERNATE_KEYS)
        assert len(traj) == 2
        assert traj.tool_names == ["search", "book"]
        assert traj.calls[0].arguments == {"query": "flights"}

    def test_with_model_and_provider(self):
        traj = generic_extract(GENERIC_TOOL_CALLS, model="gpt-4o", provider="custom")
        assert traj.model == "gpt-4o"
        assert traj.provider == "custom"
        assert all(c.model == "gpt-4o" for c in traj.calls)

    def test_empty_list(self):
        traj = generic_extract([])
        assert traj.is_empty

    def test_id_preserved(self):
        data = [{"name": "search", "arguments": {}, "id": "gen_001"}]
        traj = generic_extract(data)
        assert traj.calls[0].call_id == "gen_001"

    def test_sequential_indexing(self):
        data = [
            {"name": "a", "arguments": {}},
            {"name": "b", "arguments": {}},
            {"name": "c", "arguments": {}},
        ]
        traj = generic_extract(data)
        for expected_index, call in enumerate(traj.calls):
            assert call.call_index == expected_index
