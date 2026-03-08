"""InputSampler: abstract interface for generating varied inputs.

Behavioral assertions need N distinct inputs drawn from a defined input class.
InputSamplers produce those inputs. The three built-in strategies are:
  FixedSetSampler  - explicit list of inputs (cheapest, most predictable)
  TemplateSampler  - slot-based template expansion (structured variation)
  SemanticVariantSampler - LLM-generated phrasing variants (expensive, cached)

All samplers produce a list of InputItem, each containing the prompt string
and optional messages array for multi-turn scenarios.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from verdict.sampling.seed import SeedManager


@dataclass(frozen=True)
class InputItem:
    """A single input for a behavioral assertion trial.

    Wraps a prompt string and optional messages array so samplers
    can produce both single-turn and multi-turn inputs.
    """

    prompt: str
    messages: list[dict[str, str]] | None = None


class BaseSampler(ABC):
    """Interface for all input sampling strategies."""

    @abstractmethod
    def sample(self, n: int, seed_manager: SeedManager | None = None) -> list[InputItem]:
        """Generate n input items for behavioral assertion trials.

        Args:
            n: Number of inputs to produce.
            seed_manager: Optional SeedManager for reproducible sampling.

        Returns:
            List of InputItem, length exactly n.
        """
        ...
