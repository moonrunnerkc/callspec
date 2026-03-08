"""Verdict: top-level class that ties providers, config, and assertions together.

This is the entry point developers interact with. A Verdict instance is
constructed with a provider and optional config, then used to build
assertion chains via assert_that().
"""

from __future__ import annotations

from typing import Dict, List, Optional

from verdict.core.builder import AssertionBuilder
from verdict.core.config import VerdictConfig
from verdict.core.runner import AssertionRunner
from verdict.providers.base import BaseProvider


class Verdict:
    """The top-level object for behavioral assertion testing.

    Takes a configured provider instance and optional config. Returns
    a runner that builds assertion chains via the fluent assert_that() API.

    Usage:
        v = Verdict(provider)
        result = v.assert_that("Summarize this").is_valid_json().run()
        assert result.passed
    """

    def __init__(
        self,
        provider: BaseProvider,
        config: Optional[VerdictConfig] = None,
    ) -> None:
        self._config = config or VerdictConfig()
        self._runner = AssertionRunner(provider=provider, config=self._config)

    @property
    def config(self) -> VerdictConfig:
        return self._config

    @property
    def provider(self) -> BaseProvider:
        return self._runner.provider

    def assert_that(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> AssertionBuilder:
        """Entry point for building an assertion chain.

        The provider call is deferred until .run() is called on the returned
        builder, so constructing the chain has zero cost.
        """
        return AssertionBuilder(
            runner=self._runner,
            prompt=prompt,
            messages=messages,
        )
