"""Global configuration for Callspec assertion runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Provider retry policy: retries on network/rate-limit errors, not assertion failures.
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 1.0

# Regression drift: 15% semantic distance is the default ceiling before
# a baseline comparison fails.
REGRESSION_DRIFT_CEILING = 0.15

# Regression structural + semantic combined threshold.
REGRESSION_SEMANTIC_THRESHOLD = 0.85


@dataclass
class CallspecConfig:
    """Central configuration object threaded through the entire run.

    Every numeric constant with behavioral meaning is configurable here
    rather than scattered across assertion implementations. Defaults are
    calibrated and documented; override only with justification.
    """

    # Regression
    regression_semantic_threshold: float = REGRESSION_SEMANTIC_THRESHOLD
    regression_drift_ceiling: float = REGRESSION_DRIFT_CEILING

    # Provider
    max_retries: int = MAX_RETRIES
    retry_backoff_base_seconds: float = RETRY_BACKOFF_BASE_SECONDS
    temperature: float = 0.0
    seed: int | None = 42

    # Execution
    fail_fast: bool = True
    strict_mode: bool = False

    # Extensibility: provider-specific or assertion-specific overrides
    extra: dict[str, Any] = field(default_factory=dict)
