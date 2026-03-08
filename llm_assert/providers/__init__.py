"""Provider adapters for LLM APIs.

All provider-specific adapters use lazy imports so the core library has zero
provider dependencies. Import a provider class only when you need it; the
import will fail with an actionable error message if the corresponding
optional extra is not installed.

Always-available:
    BaseProvider    - abstract interface all providers implement
    MockProvider    - deterministic function-based provider for testing
    NormalizedResponse / ProviderResponse - common response type

Require optional extras:
    OpenAIProvider      - pip install llm-assert[openai]
    AnthropicProvider   - pip install llm-assert[anthropic]
    GoogleProvider      - pip install llm-assert[google]
    MistralProvider     - pip install llm-assert[mistral]
    OllamaProvider      - pip install llm-assert[ollama]
    LiteLLMProvider     - pip install llm-assert[litellm]
"""

from llm_assert.providers.base import BaseProvider
from llm_assert.providers.mock import MockProvider
from llm_assert.providers.response import NormalizedResponse

__all__ = [
    "BaseProvider",
    "MockProvider",
    "NormalizedResponse",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MistralProvider",
    "OllamaProvider",
    "LiteLLMProvider",
]


def __getattr__(name: str):
    """Lazy-load provider classes to avoid importing SDK packages at module level."""
    _provider_map = {
        "OpenAIProvider": "llm_assert.providers.openai",
        "AnthropicProvider": "llm_assert.providers.anthropic",
        "GoogleProvider": "llm_assert.providers.google",
        "MistralProvider": "llm_assert.providers.mistral",
        "OllamaProvider": "llm_assert.providers.ollama",
        "LiteLLMProvider": "llm_assert.providers.litellm",
    }

    if name in _provider_map:
        import importlib
        module = importlib.import_module(_provider_map[name])
        return getattr(module, name)

    raise AttributeError(f"module 'llm_assert.providers' has no attribute {name!r}")
