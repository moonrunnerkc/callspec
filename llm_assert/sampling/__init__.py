"""Input sampling for behavioral assertions.

Samplers produce varied inputs for multi-trial behavioral tests.
SeedManager provides deterministic randomness for reproducible runs.
"""

from llm_assert.sampling.sampler import BaseSampler, InputItem
from llm_assert.sampling.seed import SeedManager
from llm_assert.sampling.strategies import (
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
