"""LLMAssert pytest conftest: provides configured provider fixtures.

This conftest sets up LLMAssert providers for benchmark tests.
Provider selection is controlled by environment variables or
pytest command-line options.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-provider",
        default="openai",
        choices=["openai", "anthropic"],
        help="Provider to use for benchmark tests.",
    )


@pytest.fixture(scope="session")
def benchmark_provider(request):
    """Provide a configured LLMAssert provider based on CLI or env."""
    provider_name = request.config.getoption("--benchmark-provider")

    if provider_name == "openai":
        from llm_assert.providers.openai import OpenAIProvider
        return OpenAIProvider(
            model="gpt-4o-2024-11-20",
            temperature=0.0,
            seed=42,
        )
    elif provider_name == "anthropic":
        from llm_assert.providers.anthropic import AnthropicProvider
        return AnthropicProvider(
            model="claude-sonnet-4-20250514",
            temperature=0.0,
        )

    raise ValueError(f"Unsupported benchmark provider: {provider_name}")
