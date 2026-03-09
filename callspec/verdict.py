"""Callspec: top-level class that ties providers, config, and assertions together.

This is the entry point developers interact with. A Callspec instance is
constructed with a provider and optional config, then used to build
assertion chains via assert_that().
"""

from __future__ import annotations

from callspec.core.builder import AssertionBuilder
from callspec.core.config import CallspecConfig
from callspec.core.runner import AssertionRunner
from callspec.core.trajectory import ToolCallTrajectory
from callspec.core.trajectory_builder import TrajectoryBuilder
from callspec.providers.base import BaseProvider


class Callspec:
    """The top-level object for behavioral assertion testing.

    Takes a configured provider instance and optional config. Returns
    a runner that builds assertion chains via the fluent assert_that() API
    or trajectory assertion chains via assert_trajectory().

    Usage (content assertions):
        v = Callspec(provider)
        result = v.assert_that("Summarize this").is_valid_json().run()
        assert result.passed

    Usage (trajectory assertions):
        v = Callspec(provider)
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
        config: CallspecConfig | None = None,
    ) -> None:
        self._config = config or CallspecConfig()
        self._runner = AssertionRunner(provider=provider, config=self._config)

    @property
    def config(self) -> CallspecConfig:
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
