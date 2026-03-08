"""Recorded provider response fixtures for offline integration testing.

These fixtures contain realistic response shapes from each provider SDK,
allowing integration tests to validate response normalization and adapter
logic without making real API calls. Each fixture mirrors the exact structure
returned by the provider's Python SDK.
"""

from __future__ import annotations

from typing import Any, Dict


# OpenAI chat.completions.create response shape (SDK v1+)
OPENAI_CHAT_RESPONSE: Dict[str, Any] = {
    "id": "chatcmpl-abc123def456",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-2024-11-20",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    },
    "system_fingerprint": "fp_abc123",
}


# Anthropic messages.create response shape (SDK >=0.18.0)
ANTHROPIC_MESSAGES_RESPONSE: Dict[str, Any] = {
    "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "text",
            "text": "The capital of France is Paris.",
        }
    ],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 15,
        "output_tokens": 9,
    },
}


# Ollama chat response shape
OLLAMA_CHAT_RESPONSE: Dict[str, Any] = {
    "model": "llama3:latest",
    "created_at": "2024-01-15T10:30:00Z",
    "message": {
        "role": "assistant",
        "content": "The capital of France is Paris.",
    },
    "done": True,
    "done_reason": "stop",
    "total_duration": 1500000000,
    "load_duration": 500000000,
    "prompt_eval_count": 14,
    "prompt_eval_duration": 200000000,
    "eval_count": 8,
    "eval_duration": 800000000,
}


# Google generativeai response shape (simplified for fixture purposes)
GOOGLE_GENERATE_RESPONSE: Dict[str, Any] = {
    "model": "gemini-2.0-flash",
    "text": "The capital of France is Paris.",
    "candidates": [
        {
            "content": {
                "parts": [{"text": "The capital of France is Paris."}],
                "role": "model",
            },
            "finish_reason": "STOP",
            "safety_ratings": [],
        }
    ],
    "usage_metadata": {
        "prompt_token_count": 10,
        "candidates_token_count": 8,
        "total_token_count": 18,
    },
}


# Mistral chat.complete response shape
MISTRAL_CHAT_RESPONSE: Dict[str, Any] = {
    "id": "cmpl-e5cc70bb28c34",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "mistral-large-latest",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 8,
        "total_tokens": 19,
    },
}


# LiteLLM completion response shape (mirrors OpenAI)
LITELLM_COMPLETION_RESPONSE: Dict[str, Any] = {
    "id": "chatcmpl-litellm-xyz789",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o-2024-11-20",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    },
}
