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
    OpenAIProvider      - pip install callspec[openai]
    AnthropicProvider   - pip install callspec[anthropic]
    GoogleProvider      - pip install callspec[google]
    MistralProvider     - pip install callspec[mistral]
    OllamaProvider      - pip install callspec[ollama]
    LiteLLMProvider     - pip install callspec[litellm]
"""

from callspec.providers.base import BaseProvider
from callspec.providers.mock import MockProvider
from callspec.providers.response import NormalizedResponse

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
        "OpenAIProvider": "callspec.providers.openai",
        "AnthropicProvider": "callspec.providers.anthropic",
        "GoogleProvider": "callspec.providers.google",
        "MistralProvider": "callspec.providers.mistral",
        "OllamaProvider": "callspec.providers.ollama",
        "LiteLLMProvider": "callspec.providers.litellm",
    }

    if name in _provider_map:
        import importlib
        module = importlib.import_module(_provider_map[name])
        return getattr(module, name)

    raise AttributeError(f"module 'callspec.providers' has no attribute {name!r}")
