"""NormalizedResponse: re-export from core types for convenience.

Provider adapters import from here; internal code imports from core.types.
This avoids circular imports while keeping the provider package self-contained.
"""

from verdict.core.types import ProviderResponse as NormalizedResponse

__all__ = ["NormalizedResponse"]
