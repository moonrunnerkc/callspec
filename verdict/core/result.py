"""AssertionResult and SuiteResult are defined in core.types.

This module re-exports them for import convenience and provides
any result-level utility functions.
"""

from verdict.core.types import AssertionResult, IndividualAssertionResult, SuiteResult

__all__ = ["AssertionResult", "IndividualAssertionResult", "SuiteResult"]
