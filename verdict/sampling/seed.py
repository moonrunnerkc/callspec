"""SeedManager: deterministic seed control for reproducible behavioral test runs.

Behavioral assertions run the provider N times. Without controlled seeding,
the input ordering and any randomized sampling would produce different runs
each time, undermining reproducibility. The SeedManager produces a deterministic
sequence of seeds derived from a single root seed, so the same root seed
always produces the same sequence of provider calls and input selections.

This is separate from the provider-level seed (which controls the LLM's
own randomness). The SeedManager controls Verdict's sampling randomness.
"""

from __future__ import annotations

import random
from typing import Optional


class SeedManager:
    """Produces deterministic random state from a single root seed.

    Each call to next_seed() returns the next integer in a deterministic
    sequence. The sequence is fully determined by the root seed,
    so behavioral test runs are reproducible regardless of execution order.
    """

    def __init__(self, root_seed: Optional[int] = 42) -> None:
        self._root_seed = root_seed
        self._rng = random.Random(root_seed)

    @property
    def root_seed(self) -> Optional[int]:
        return self._root_seed

    def next_seed(self) -> int:
        """Return the next deterministic seed in the sequence."""
        return self._rng.randint(0, 2**31 - 1)

    def get_rng(self) -> random.Random:
        """Return a new Random instance seeded from the next seed.

        Useful when a component needs its own independent random state
        without perturbing the main sequence for other components.
        """
        return random.Random(self.next_seed())

    def reset(self) -> None:
        """Reset the sequence to the beginning.

        After reset, the same sequence of seeds will be produced.
        Used between test suite runs for reproducibility.
        """
        self._rng = random.Random(self._root_seed)
