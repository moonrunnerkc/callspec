"""Shared types used across the Callspec library."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AssertionType(Enum):
    """Categories of assertions supported by callspec."""

    STRUCTURAL = "structural"
    REGRESSION = "regression"


class Severity(Enum):
    """Controls whether a failing assertion fails the suite or produces a warning."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ProviderResponse:
    """Normalized structure returned by all provider adapters.

    Every field beyond `content` is optional because not all providers
    expose the same metadata. The `raw` dict preserves the provider's
    original response for debugging without forcing normalization of
    provider-specific fields.
    """

    content: str
    raw: dict[str, Any] = field(default_factory=dict)
    model: str = "unknown"
    provider: str = "unknown"
    latency_ms: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    request_id: str | None = None
    # Raw tool call data extracted by the provider adapter. Each entry is a
    # dict with at minimum {"name": str, "arguments": dict}. Provider-specific
    # fields (id, call_id) are preserved for traceability.
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IndividualAssertionResult:
    """Result of a single assertion within a chain.

    Every field that matters for debugging a failure at 2am is included:
    the type, the name, the score, the threshold, and a human-readable
    message explaining exactly what happened.
    """

    assertion_type: str
    assertion_name: str
    passed: bool
    message: str
    score: float | None = None
    threshold: float | None = None
    confidence: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssertionResult:
    """Aggregate result of running an assertion chain against a provider.

    `passed` is True only when every individual assertion in the chain passed.
    The full list of individual results is always available regardless of
    fail-fast behavior, so a developer can inspect partial results after
    an early exit.
    """

    passed: bool
    assertions: list[IndividualAssertionResult]
    provider_response: ProviderResponse | None = None
    execution_time_ms: int = 0
    model: str = "unknown"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class SuiteResult:
    """Aggregate result of running a full assertion suite.

    Contains individual AssertionResults keyed by case name, plus
    suite-level metadata for reporting.
    """

    passed: bool
    case_results: dict[str, AssertionResult] = field(default_factory=dict)
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    execution_time_ms: int = 0
