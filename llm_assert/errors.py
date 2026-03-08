"""All Verdict exception types.

Each exception carries enough context for a developer debugging at 2am
to understand what failed, where, and what to try next. Generic messages
like "assertion failed" or "provider error" are not acceptable here.
"""

from __future__ import annotations

from typing import Any


class VerdictError(Exception):
    """Base exception for all Verdict errors."""

    pass


class ProviderError(VerdictError):
    """Raised when a provider call fails after exhausting retries.

    Includes the provider name, the number of attempts, and the
    underlying cause so the developer knows whether to check credentials,
    network, or rate limits.
    """

    def __init__(
        self,
        provider: str,
        message: str,
        attempts: int = 1,
        cause: Exception | None = None,
    ) -> None:
        self.provider = provider
        self.attempts = attempts
        self.cause = cause
        full_message = (
            f"ProviderError [{provider}]: {message} "
            f"(after {attempts} attempt{'s' if attempts != 1 else ''})"
        )
        if cause:
            full_message += f". Cause: {type(cause).__name__}: {cause}"
        super().__init__(full_message)


class ProviderNotConfiguredError(VerdictError):
    """Raised when a provider is referenced but not installed or configured."""

    def __init__(self, provider: str, install_hint: str = "") -> None:
        self.provider = provider
        message = f"Provider '{provider}' is not configured or not installed."
        if install_hint:
            message += f" Install with: {install_hint}"
        super().__init__(message)


class AssertionError(VerdictError):
    """Raised when an assertion encounters an internal error during evaluation.

    This is not a test failure (those produce AssertionResult with passed=False).
    This is an error in the assertion logic itself, such as an invalid schema
    or a misconfigured threshold.
    """

    def __init__(
        self,
        assertion_name: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.assertion_name = assertion_name
        self.details = details or {}
        full_message = f"AssertionError [{assertion_name}]: {message}"
        super().__init__(full_message)


class SnapshotError(VerdictError):
    """Raised on snapshot lifecycle failures: missing baseline, corrupt file, schema mismatch."""

    def __init__(self, snapshot_key: str, message: str) -> None:
        self.snapshot_key = snapshot_key
        super().__init__(f"SnapshotError [{snapshot_key}]: {message}")


class ConfigurationError(VerdictError):
    """Raised when VerdictConfig or YAML suite config is invalid."""

    def __init__(self, message: str, field: str | None = None) -> None:
        self.field = field
        prefix = f"ConfigurationError [{field}]" if field else "ConfigurationError"
        super().__init__(f"{prefix}: {message}")


class SuiteParseError(VerdictError):
    """Raised when a YAML suite file cannot be parsed or validated.

    Includes the line number when available so the developer can jump
    directly to the problem.
    """

    def __init__(
        self,
        filepath: str,
        message: str,
        line_number: int | None = None,
    ) -> None:
        self.filepath = filepath
        self.line_number = line_number
        location = f"{filepath}:{line_number}" if line_number else filepath
        super().__init__(f"SuiteParseError [{location}]: {message}")
