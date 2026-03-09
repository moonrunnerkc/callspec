"""Recorded response fixtures for tool-call capture testing.

Real response structures from each provider format, including edge cases
like parallel tool calls, legacy function_call, and empty tool calls.
These fixtures are the source of truth for capture adapter correctness.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# OpenAI Chat Completions: single tool call
# ---------------------------------------------------------------------------
OPENAI_CHAT_SINGLE_TOOL_CALL: dict = {
    "id": "chatcmpl-abc123",
    "model": "gpt-4o-2024-11-20",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz789",
                        "type": "function",
                        "function": {
                            "name": "search_flights",
                            "arguments": '{"origin": "NYC", "destination": "London", "date": "2026-03-15"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 42, "completion_tokens": 28, "total_tokens": 70},
}

# ---------------------------------------------------------------------------
# OpenAI Chat Completions: parallel tool calls (multiple in one message)
# ---------------------------------------------------------------------------
OPENAI_CHAT_PARALLEL_TOOL_CALLS: dict = {
    "id": "chatcmpl-def456",
    "model": "gpt-4o-2024-11-20",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_aaa",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "London"}',
                        },
                    },
                    {
                        "id": "call_bbb",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    },
                    {
                        "id": "call_ccc",
                        "type": "function",
                        "function": {
                            "name": "convert_currency",
                            "arguments": '{"from": "USD", "to": "EUR", "amount": 100}',
                        },
                    },
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 55, "completion_tokens": 45, "total_tokens": 100},
}

# ---------------------------------------------------------------------------
# OpenAI Chat Completions: legacy function_call format (deprecated)
# ---------------------------------------------------------------------------
OPENAI_CHAT_LEGACY_FUNCTION_CALL: dict = {
    "id": "chatcmpl-ghi789",
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_user_info",
                    "arguments": '{"user_id": "u_12345"}',
                },
            },
            "finish_reason": "function_call",
        }
    ],
    "usage": {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45},
}

# ---------------------------------------------------------------------------
# OpenAI Chat Completions: text-only response (no tool calls)
# ---------------------------------------------------------------------------
OPENAI_CHAT_NO_TOOL_CALLS: dict = {
    "id": "chatcmpl-jkl012",
    "model": "gpt-4o-2024-11-20",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The weather in London is sunny today.",
                "tool_calls": None,
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}

# ---------------------------------------------------------------------------
# OpenAI Responses API format (March 2025+)
# ---------------------------------------------------------------------------
OPENAI_RESPONSES_API_SINGLE: dict = {
    "id": "resp_abc123",
    "model": "gpt-4o-2025-03-01",
    "output": [
        {
            "type": "function_call",
            "call_id": "fc_xyz789",
            "name": "search_flights",
            "arguments": '{"origin": "NYC", "destination": "London"}',
        }
    ],
}

OPENAI_RESPONSES_API_MULTIPLE: dict = {
    "id": "resp_def456",
    "model": "gpt-4o-2025-03-01",
    "output": [
        {
            "type": "message",
            "content": [{"type": "text", "text": "Let me search for flights."}],
        },
        {
            "type": "function_call",
            "call_id": "fc_aaa",
            "name": "search_flights",
            "arguments": '{"origin": "NYC", "destination": "London"}',
        },
        {
            "type": "function_call",
            "call_id": "fc_bbb",
            "name": "check_availability",
            "arguments": '{"flight_id": "BA123", "date": "2026-03-15"}',
        },
    ],
}

# ---------------------------------------------------------------------------
# OpenAI: malformed arguments (JSON parse error)
# ---------------------------------------------------------------------------
OPENAI_CHAT_MALFORMED_ARGS: dict = {
    "id": "chatcmpl-bad",
    "model": "gpt-4o-2024-11-20",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {
                            "name": "broken_tool",
                            "arguments": "not valid json {{{",
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
}

# ---------------------------------------------------------------------------
# Anthropic: single tool_use block
# ---------------------------------------------------------------------------
ANTHROPIC_SINGLE_TOOL_USE: dict = {
    "id": "msg_abc123",
    "type": "message",
    "model": "claude-sonnet-4-20250514",
    "content": [
        {"type": "text", "text": "I'll search for flights for you."},
        {
            "type": "tool_use",
            "id": "toolu_xyz789",
            "name": "search_flights",
            "input": {"origin": "NYC", "destination": "London", "date": "2026-03-15"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 50, "output_tokens": 35},
}

# ---------------------------------------------------------------------------
# Anthropic: multiple tool_use blocks interleaved with text
# ---------------------------------------------------------------------------
ANTHROPIC_MULTIPLE_TOOL_USE: dict = {
    "id": "msg_def456",
    "type": "message",
    "model": "claude-sonnet-4-20250514",
    "content": [
        {"type": "text", "text": "Let me check both options."},
        {
            "type": "tool_use",
            "id": "toolu_aaa",
            "name": "search_flights",
            "input": {"origin": "NYC", "destination": "London"},
        },
        {"type": "text", "text": "Now checking hotels..."},
        {
            "type": "tool_use",
            "id": "toolu_bbb",
            "name": "search_hotels",
            "input": {"city": "London", "checkin": "2026-03-15"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 60, "output_tokens": 50},
}

# ---------------------------------------------------------------------------
# Anthropic: text-only (no tool calls)
# ---------------------------------------------------------------------------
ANTHROPIC_NO_TOOL_USE: dict = {
    "id": "msg_ghi789",
    "type": "message",
    "model": "claude-sonnet-4-20250514",
    "content": [
        {"type": "text", "text": "The weather in London is quite pleasant in March."},
    ],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 25, "output_tokens": 15},
}

# ---------------------------------------------------------------------------
# Generic: plain list of tool-call dicts
# ---------------------------------------------------------------------------
GENERIC_TOOL_CALLS: list = [
    {"name": "search", "arguments": {"query": "flights to London"}},
    {"name": "book", "arguments": {"flight_id": "BA123", "passenger": "Brad"}},
]

GENERIC_TOOL_CALLS_ALTERNATE_KEYS: list = [
    {"tool_name": "search", "args": {"query": "flights"}},
    {"tool_name": "book", "args": {"flight_id": "BA123"}},
]
