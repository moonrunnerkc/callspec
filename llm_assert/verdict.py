"""LLMAssert: top-level class that ties providers, config, and assertions together.

This is the entry point developers interact with. A LLMAssert instance is
constructed with a provider and optional config, then used to build
assertion chains via assert_that().
"""

from __future__ import annotations

from llm_assert.core.builder import AssertionBuilder
from llm_assert.core.config import LLMAssertConfig
from llm_assert.core.runner import AssertionRunner
from llm_assert.core.trajectory import ToolCallTrajectory
from llm_assert.core.trajectory_builder import TrajectoryBuilder
from llm_assert.providers.base import BaseProvider


class LLMAssert:
    """The top-level object for behavioral assertion testing.

    Takes a configured provider instance and optional config. Returns
    a runner that builds assertion chains via the fluent assert_that() API
    or trajectory assertion chains via assert_trajectory().

    Usage (content assertions):
        v = LLMAssert(provider)
        result = v.assert_that("Summarize this").is_valid_json().run()
        assert result.passed

    Usage (trajectory assertions):
        v = LLMAssert(provider)
        trajectory = ToolCallTrajectory(calls=[...])
        result = (
            v.assert_trajectory(trajectory)
            .calls_tools_in_order(["search", "book"])
            .argument_not_empty("search", "query")
            .run()
        )
        assert result.passed
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: LLMAssertConfig | None = None,
    ) -> None:
        self._config = config or LLMAssertConfig()
        self._runner = AssertionRunner(provider=provider, config=self._config)

    @property
    def config(self) -> LLMAssertConfig:
        return self._config

    @property
    def provider(self) -> BaseProvider:
        return self._runner.provider

    def assert_that(
        self,
        prompt: str,
        messages: list[dict[str, str]] | None = None,
    ) -> AssertionBuilder:
        """Entry point for building a content assertion chain.

        The provider call is deferred until .run() is called on the returned
        builder, so constructing the chain has zero cost.
        """
        return AssertionBuilder(
            runner=self._runner,
            prompt=prompt,
            messages=messages,
        )

    def assert_trajectory(
        self,
        trajectory: ToolCallTrajectory,
    ) -> TrajectoryBuilder:
        """Entry point for building a trajectory assertion chain.

        The trajectory is already captured; no provider call needed.
        Chain trajectory and contract assertions, then call .run().
        """
        return TrajectoryBuilder(
            trajectory=trajectory,
            config=self._config,
        )
