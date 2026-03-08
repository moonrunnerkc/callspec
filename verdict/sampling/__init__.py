"""Input sampling for behavioral assertions.

Samplers produce varied inputs for multi-trial behavioral tests.
SeedManager provides deterministic randomness for reproducible runs.
"""

from verdict.sampling.sampler import BaseSampler, InputItem
from verdict.sampling.seed import SeedManager
from verdict.sampling.strategies import (
    FixedSetSampler,
    SemanticVariantSampler,
    TemplateSampler,
)

__all__ = [
    "BaseSampler",
    "InputItem",
    "SeedManager",
    "FixedSetSampler",
    "TemplateSampler",
    "SemanticVariantSampler",
]
