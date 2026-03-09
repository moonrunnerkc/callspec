"""TrajectoryAssertion: base interface for assertions over tool-call trajectories.

Parallels BaseAssertion but operates on ToolCallTrajectory objects instead
of content strings. Trajectory assertions verify the shape and ordering of
tool calls. Contract assertions (a subtype) verify individual tool call
arguments. Both extend this base.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from llm_assert.core.config import LLMAssertConfig
from llm_assert.core.trajectory import ToolCallTrajectory
from llm_assert.core.types import IndividualAssertionResult


class TrajectoryAssertion(ABC):
    """Interface for assertions that evaluate a ToolCallTrajectory.

    Subclasses implement `evaluate_trajectory`, which receives the
    trajectory and returns a structured result. The assertion_type
    is "trajectory" for sequence-level checks and "contract" for
    per-argument checks.
    """

    assertion_type: str = "trajectory"
    assertion_name: str = "unknown"

    @abstractmethod
    def evaluate_trajectory(
        self,
        trajectory: ToolCallTrajectory,
        config: LLMAssertConfig,
    ) -> IndividualAssertionResult:
        """Evaluate this assertion against the given trajectory.

        Args:
            trajectory: The tool-call sequence to evaluate.
            config: The active LLMAssertConfig for behavior overrides.

        Returns:
            IndividualAssertionResult with pass/fail and diagnostic message.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(type={self.assertion_type}, name={self.assertion_name})"
