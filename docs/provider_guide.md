# Provider Guide

Verdict works with any LLM provider through a thin adapter layer. Assertions are provider-agnostic: the same assertion chain works identically whether you use OpenAI, Anthropic, a local Ollama model, or a mock.

## Supported Providers

| Provider | Install | Default Model | API Key Env Var |
|----------|---------|---------------|-----------------|
| OpenAI | `verdict[openai]` | `gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `verdict[anthropic]` | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| Google | `verdict[google]` | `gemini-2.0-flash` | `GOOGLE_API_KEY` |
| Mistral | `verdict[mistral]` | `mistral-large-latest` | (via SDK) |
| Ollama | `verdict[ollama]` | `llama3` | None (local) |
| LiteLLM | `verdict[litellm]` | `gpt-4o` | (per-provider) |
| Mock | (built-in) | `mock` | None |

## OpenAI

```bash
pip install "verdict[openai]"
export OPENAI_API_KEY="sk-..."
```

```python
from verdict.providers.openai import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o")
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4o"` | Model identifier |
| `api_key` | `str` or `None` | `None` | Falls back to `OPENAI_API_KEY` env var |
| `base_url` | `str` or `None` | `None` | Custom API endpoint (for proxies or Azure) |
| `organization` | `str` or `None` | `None` | OpenAI organization ID |
| `temperature` | `float` | `0.0` | Generation temperature |
| `seed` | `int` | `42` | Deterministic seed (supported since Nov 2023) |
| `max_tokens` | `int` or `None` | `None` | Maximum output tokens |

OpenAI supports deterministic output via the `seed` parameter. At `temperature=0` with a fixed seed, outputs are reproducible.

### Azure OpenAI

Use `base_url` for Azure endpoints:

```python
provider = OpenAIProvider(
    model="my-deployment-name",
    base_url="https://my-resource.openai.azure.com/openai/deployments/my-deployment",
    api_key="azure-key-here",
)
```

## Anthropic

```bash
pip install "verdict[anthropic]"
export ANTHROPIC_API_KEY="sk-ant-..."
```

```python
from verdict.providers.anthropic import AnthropicProvider

provider = AnthropicProvider(model="claude-sonnet-4-20250514")
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model identifier |
| `api_key` | `str` or `None` | `None` | Falls back to `ANTHROPIC_API_KEY` env var |
| `temperature` | `float` | `0.0` | Generation temperature |
| `max_tokens` | `int` | `1024` | Required by Anthropic API |

Anthropic does not offer a `seed` parameter. At `temperature=0`, Claude's outputs are highly consistent but not perfectly deterministic. Tests using Anthropic may produce occasional variance; the confidence interval mechanism in behavioral assertions accounts for this.

The adapter automatically separates the `system` role from messages into Anthropic's top-level `system` parameter.

## Google Generative AI

```bash
pip install "verdict[google]"
export GOOGLE_API_KEY="AIza..."
```

```python
from verdict.providers.google import GoogleProvider

provider = GoogleProvider(model="gemini-2.0-flash")
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gemini-2.0-flash"` | Model identifier |
| `api_key` | `str` or `None` | `None` | Falls back to `GOOGLE_API_KEY` env var |
| `temperature` | `float` | `0.0` | Generation temperature |
| `max_output_tokens` | `int` or `None` | `None` | Maximum output tokens |

The adapter maps OpenAI-style roles to Google roles (`"assistant"` becomes `"model"`, `"system"` is prepended to the first user message).

## Mistral

```bash
pip install "verdict[mistral]"
```

```python
from verdict.providers.mistral import MistralProvider

provider = MistralProvider(model="mistral-large-latest")
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"mistral-large-latest"` | Model identifier |
| `api_key` | `str` or `None` | `None` | Falls back to SDK default |
| `temperature` | `float` | `0.0` | Generation temperature |
| `max_tokens` | `int` or `None` | `None` | Maximum output tokens |

Mistral does not expose a seed parameter.

## Ollama (Local)

```bash
pip install "verdict[ollama]"
# Ollama server must be running locally
ollama pull llama3
```

```python
from verdict.providers.ollama import OllamaProvider

provider = OllamaProvider(model="llama3")
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"llama3"` | Model name (must be pulled locally) |
| `host` | `str` or `None` | `None` | Ollama server URL (default: `http://localhost:11434`) |
| `temperature` | `float` | `0.0` | Generation temperature |
| `seed` | `int` | `42` | Deterministic seed |
| `num_predict` | `int` or `None` | `None` | Maximum output tokens |

No API key required. The model runs locally on your machine. Supports deterministic output via `seed`.

## LiteLLM (Universal Adapter)

LiteLLM routes to any provider via model string prefixes. Use this when you want a single adapter for multiple providers or when your provider is not directly supported.

```bash
pip install "verdict[litellm]"
```

```python
from verdict.providers.litellm import LiteLLMProvider

# OpenAI via LiteLLM
provider = LiteLLMProvider(model="gpt-4o", api_key="sk-...")

# Anthropic via LiteLLM
provider = LiteLLMProvider(model="anthropic/claude-sonnet-4-20250514", api_key="sk-ant-...")

# Any provider LiteLLM supports
provider = LiteLLMProvider(model="together_ai/meta-llama/Llama-3-70b", api_key="...")
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-4o"` | Model string with optional provider prefix |
| `api_key` | `str` or `None` | `None` | API key for the underlying provider |
| `api_base` | `str` or `None` | `None` | Custom API base URL |
| `temperature` | `float` | `0.0` | Generation temperature |
| `seed` | `int` | `42` | Deterministic seed (when supported) |
| `max_tokens` | `int` or `None` | `None` | Maximum output tokens |

## Mock Provider

The mock provider returns deterministic responses without network calls. Use it for testing assertion configurations, unit tests, and CI environments without API spend.

```python
from verdict.providers.mock import MockProvider

# Simple string response
provider = MockProvider(
    lambda prompt, messages: '{"title": "Test", "summary": "A summary"}'
)

# Dynamic response based on prompt
def dynamic_response(prompt, messages=None):
    if "JSON" in prompt:
        return '{"result": "structured"}'
    return "Plain text response"

provider = MockProvider(dynamic_response)
```

**Constructor parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `response_fn` | `Callable` | | Function mapping `(prompt, messages)` to response string |
| `model_name` | `str` | `"mock"` | Model name in result metadata |
| `latency_ms` | `int` | `0` | Simulated latency (for timing tests) |

## Provider API

All providers implement the `BaseProvider` interface:

```python
class BaseProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def call(self, prompt, messages=None, **kwargs) -> ProviderResponse: ...

    async def call_async(self, prompt, messages=None, **kwargs) -> ProviderResponse: ...

    async def batch_call(self, prompts, **kwargs) -> list[ProviderResponse]: ...
```

`call_async` defaults to running `call` in a thread executor. Providers with native async support (OpenAI, Anthropic, Google, Mistral, Ollama, LiteLLM) override this with their SDK's async method.

`batch_call` runs multiple calls concurrently via `asyncio.gather`. Providers with native batch APIs can override for efficiency.

## NormalizedResponse

Every provider returns a `ProviderResponse` with a consistent structure:

```python
@dataclass(frozen=True)
class ProviderResponse:
    content: str              # The response text
    raw: dict                 # Original provider response, unmodified
    model: str                # Exact model identifier from the response
    provider: str             # Provider name
    latency_ms: int           # End-to-end call time
    prompt_tokens: int | None # Token count when available
    completion_tokens: int | None
    finish_reason: str | None # "stop", "length", "content_filter", etc.
    request_id: str | None    # For traceability
```

The `model` field contains the actual model identifier returned by the API, not the alias you requested. When OpenAI silently updates what "gpt-4o" points to, the response `model` reflects the actual model string.

## Writing a Custom Provider

Implement `BaseProvider` for any LLM service:

```python
from verdict.providers.base import BaseProvider
from verdict.core.types import ProviderResponse

class MyProvider(BaseProvider):
    @property
    def provider_name(self) -> str:
        return "my_provider"

    def call(self, prompt, messages=None, **kwargs) -> ProviderResponse:
        # Call your API here
        response = my_api.generate(prompt)
        return ProviderResponse(
            content=response.text,
            raw=response.to_dict(),
            model=response.model_id,
            provider=self.provider_name,
            latency_ms=response.latency,
        )
```

## Checking Provider Connectivity

```bash
# Check all configured providers
verdict check

# List installed providers and their status
verdict providers
```

The `verdict check` command verifies each configured provider is reachable and responds correctly. It is the first thing to run after installation.
