"""Callspec: Behavioral assertion testing for LLM applications."""

from callspec.core.config import CallspecConfig
from callspec.core.trajectory import ToolCall, ToolCallTrajectory
from callspec.core.trajectory_builder import TrajectoryBuilder
from callspec.core.types import (
    AssertionResult,
    AssertionType,
    IndividualAssertionResult,
    ProviderResponse,
    Severity,
    SuiteResult,
)
from callspec.verdict import Callspec
from callspec.version import __version__

__all__ = [
    "__version__",
    "Callspec",
    "CallspecConfig",
    "AssertionResult",
    "AssertionType",
    "IndividualAssertionResult",
    "ProviderResponse",
    "Severity",
    "SuiteResult",
    "ToolCall",
    "ToolCallTrajectory",
    "TrajectoryBuilder",
]
