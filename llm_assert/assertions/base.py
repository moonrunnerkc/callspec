"""BaseAssertion: abstract interface that all assertions implement.

Every assertion takes a string (the provider response content) and returns
an IndividualAssertionResult. The assertion does not call the provider;
the runner handles that. This separation keeps assertions pure functions
over strings, which makes them testable without any provider dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from verdict.core.config import VerdictConfig
from verdict.core.types import IndividualAssertionResult


class BaseAssertion(ABC):
    """Interface for all Verdict assertions.

    Subclasses implement `evaluate`, which receives the response content
    as a string and returns a structured result. The assertion type
    (structural, semantic, behavioral, regression) is declared by each
    subclass as a class attribute.
    """

    assertion_type: str = "unknown"
    assertion_name: str = "unknown"

    @abstractmethod
    def evaluate(self, content: str, config: VerdictConfig) -> IndividualAssertionResult:
        """Evaluate this assertion against the given content.

        Args:
            content: The provider response text to evaluate.
            config: The active VerdictConfig for threshold and behavior overrides.

        Returns:
            IndividualAssertionResult with pass/fail, score, and diagnostic message.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.assertion_type}, name={self.assertion_name})"
