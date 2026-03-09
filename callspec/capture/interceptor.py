"""Interceptor: wraps a provider call and captures the tool-call trajectory.

The interceptor is a convenience layer. Users who already have a response
object can pass it directly to the normalizer. The interceptor handles
the "call the provider and normalize the response" pattern in one step.
"""

from __future__ import annotations

from typing import Any

from callspec.capture.normalizer import normalize
from callspec.core.trajectory import ToolCallTrajectory
from callspec.core.types import ProviderResponse
from callspec.providers.base import BaseProvider


class CaptureInterceptor:
    """Wraps a provider and captures tool-call trajectories from responses.

    Usage:
        interceptor = CaptureInterceptor(openai_provider)
        trajectory = interceptor.capture("Book a flight to London")
        # trajectory is a ToolCallTrajectory ready for assertions
    """

    def __init__(self, provider: BaseProvider) -> None:
        self._provider = provider

    def capture(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ToolCallTrajectory:
        """Call the provider and return the captured tool-call trajectory.

        Passes all arguments through to the provider's call() method.
        The ProviderResponse.tool_calls field is normalized into a
        ToolCallTrajectory. If the response contains no tool calls,
        returns an empty trajectory.
        """
        response = self._provider.call(prompt, messages, **kwargs)
        return normalize(response)

    async def capture_async(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ToolCallTrajectory:
        """Async variant of capture()."""
        response = await self._provider.call_async(prompt, messages, **kwargs)
        return normalize(response)

    def call_and_capture(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> tuple[ProviderResponse, ToolCallTrajectory]:
        """Call the provider and return both the response and the trajectory.

        Useful when you need the full response for content assertions
        and the trajectory for tool-call assertions.
        """
        response = self._provider.call(prompt, messages, **kwargs)
        trajectory = normalize(response)
        return response, trajectory
