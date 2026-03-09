"""Abstract base class for all LLM provider adapters.

Every provider implements three methods: call, call_async, batch_call.
The contract is simple: take a prompt (or messages), return a ProviderResponse.
Validation happens at the boundary (here); internal code trusts the contract.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from callspec.core.types import ProviderResponse


class BaseProvider(ABC):
    """Interface that all provider adapters implement.

    Subclasses must implement `call`. Async and batch have sensible defaults
    that subclasses can override when the provider offers native support
    (e.g., Anthropic batch API, OpenAI async client).
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for this provider, used in logs and reports."""
        ...

    @abstractmethod
    def call(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Synchronous provider call. Returns a normalized response."""
        ...

    async def call_async(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Async provider call.

        Default implementation runs the sync call in a thread executor.
        Providers with native async support (OpenAI, Anthropic) override this.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.call(prompt, messages, **kwargs),
        )

    async def batch_call(
        self,
        prompts: Sequence[str],
        messages_list: Sequence[list[dict[str, str]] | None] | None = None,
        **kwargs: Any,
    ) -> list[ProviderResponse]:
        """Batched provider call for behavioral assertions.

        Default implementation uses asyncio.gather over call_async.
        Providers with native batch endpoints override this for efficiency.
        """
        if messages_list is None:
            messages_list = [None] * len(prompts)

        tasks = [
            self.call_async(prompt, messages, **kwargs)
            for prompt, messages in zip(prompts, messages_list)
        ]
        return list(await asyncio.gather(*tasks))
