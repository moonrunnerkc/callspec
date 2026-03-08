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
    OpenAIProvider      - pip install verdict[openai]
    AnthropicProvider   - pip install verdict[anthropic]
    GoogleProvider      - pip install verdict[google]
    MistralProvider     - pip install verdict[mistral]
    OllamaProvider      - pip install verdict[ollama]
    LiteLLMProvider     - pip install verdict[litellm]
"""

from verdict.providers.base import BaseProvider
from verdict.providers.mock import MockProvider
from verdict.providers.response import NormalizedResponse

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
        "OpenAIProvider": "verdict.providers.openai",
        "AnthropicProvider": "verdict.providers.anthropic",
        "GoogleProvider": "verdict.providers.google",
        "MistralProvider": "verdict.providers.mistral",
        "OllamaProvider": "verdict.providers.ollama",
        "LiteLLMProvider": "verdict.providers.litellm",
    }

    if name in _provider_map:
        import importlib
        module = importlib.import_module(_provider_map[name])
        return getattr(module, name)

    raise AttributeError(f"module 'verdict.providers' has no attribute {name!r}")