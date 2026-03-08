"""Built-in input sampling strategies for behavioral assertions.

FixedSetSampler  - explicit list, cycled or shuffled to fill N samples
TemplateSampler  - slot-based template expansion with combinatorial variants
SemanticVariantSampler - LLM-generated phrasing variants with disk caching
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llm_assert.sampling.sampler import BaseSampler, InputItem
from llm_assert.sampling.seed import SeedManager

logger = logging.getLogger(__name__)


class FixedSetSampler(BaseSampler):
    """Returns inputs from an explicit list, cycling if n exceeds the list length.

    The simplest sampler: you define the inputs, they run in order (or shuffled).
    Appropriate when the input class is small and well-defined.
    """

    def __init__(
        self,
        inputs: Sequence[str | InputItem | dict[str, Any]],
        shuffle: bool = False,
    ) -> None:
        self._inputs = self._normalize_inputs(inputs)
        self._shuffle = shuffle

    @staticmethod
    def _normalize_inputs(raw: Sequence[str | InputItem | dict[str, Any]]) -> list[InputItem]:
        """Coerce string, dict, or InputItem inputs into a uniform InputItem list."""
        normalized = []
        for entry in raw:
            if isinstance(entry, InputItem):
                normalized.append(entry)
            elif isinstance(entry, str):
                normalized.append(InputItem(prompt=entry))
            elif isinstance(entry, dict):
                normalized.append(InputItem(
                    prompt=entry.get("prompt", ""),
                    messages=entry.get("messages"),
                ))
            else:
                raise TypeError(
                    f"FixedSetSampler expects str, InputItem, "
                    f"or dict entries, got {type(entry).__name__}"
                )
        return normalized

    def sample(self, n: int, seed_manager: SeedManager | None = None) -> list[InputItem]:
        if not self._inputs:
            raise ValueError("FixedSetSampler has no inputs to sample from")

        pool = list(self._inputs)

        if self._shuffle and seed_manager is not None:
            rng = seed_manager.get_rng()
            rng.shuffle(pool)
        elif self._shuffle:
            import random
            random.shuffle(pool)

        # Cycle through inputs to fill exactly n samples
        cycled = list(itertools.islice(itertools.cycle(pool), n))
        return cycled


class TemplateSampler(BaseSampler):
    """Generates inputs by expanding a template with variable combinations.

    Takes a template string with {named_slots} and a dict mapping each
    slot name to a list of possible values. Produces inputs by combining
    slot values, either exhaustively (all combinations up to n) or
    randomly sampled.

    Example:
        template = "What is the capital of {country}?"
        variables = {"country": ["France", "Germany", "Japan"]}
        # Produces: "What is the capital of France?", etc.
    """

    def __init__(
        self,
        template: str,
        variables: dict[str, list[str]],
        exhaustive: bool = True,
    ) -> None:
        self._template = template
        self._variables = variables
        self._exhaustive = exhaustive

        # Validate that all template slots have corresponding variable lists
        import re
        slot_names = set(re.findall(r"\{(\w+)\}", template))
        missing_slots = slot_names - set(variables.keys())
        if missing_slots:
            raise ValueError(
                f"Template contains slots without variable lists: {missing_slots}. "
                f"Available variables: {set(variables.keys())}"
            )

    def _generate_all_combinations(self) -> list[str]:
        """Produce all possible template expansions from the variable lists."""
        slot_names = sorted(self._variables.keys())
        value_lists = [self._variables[name] for name in slot_names]

        expanded = []
        for combination in itertools.product(*value_lists):
            mapping = dict(zip(slot_names, combination))
            expanded.append(self._template.format(**mapping))

        return expanded

    def sample(self, n: int, seed_manager: SeedManager | None = None) -> list[InputItem]:
        all_expansions = self._generate_all_combinations()

        if not all_expansions:
            raise ValueError("TemplateSampler produced no expansions; check variable lists")

        if self._exhaustive:
            # Use all combinations, cycling if n exceeds the total count
            pool = all_expansions
        else:
            pool = list(all_expansions)

        if not self._exhaustive and seed_manager is not None:
            rng = seed_manager.get_rng()
            rng.shuffle(pool)
        elif not self._exhaustive:
            import random
            random.shuffle(pool)

        cycled = list(itertools.islice(itertools.cycle(pool), n))
        return [InputItem(prompt=text) for text in cycled]


class SemanticVariantSampler(BaseSampler):
    """Generates semantically similar but differently-phrased versions of a seed input.

    Uses an LLM to produce phrasing variants, then caches them to disk so
    subsequent runs do not regenerate (avoiding API cost on every test run).

    The cache key is derived from the seed input content and n_variants,
    so changing either invalidates the cache.

    Cost: n_variants LLM calls on first run, zero on subsequent runs.
    The architecture doc recommends starting with FixedSetSampler until
    the assertion configuration is stable before switching to this sampler.
    """

    # Directory for cached variant files, relative to the project root
    DEFAULT_CACHE_DIR = ".llm_assert_cache/semantic_variants"

    def __init__(
        self,
        seed_input: str,
        n_variants: int = 20,
        provider: Any = None,
        cache_dir: str | None = None,
    ) -> None:
        self._seed_input = seed_input
        self._n_variants = n_variants
        self._provider = provider
        self._cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)

    def _cache_key(self) -> str:
        """Deterministic cache key from seed input and variant count."""
        content_hash = hashlib.sha256(
            f"{self._seed_input}::{self._n_variants}".encode()
        ).hexdigest()[:16]
        return f"variants_{content_hash}"

    def _cache_path(self) -> Path:
        return self._cache_dir / f"{self._cache_key()}.json"

    def _load_cached_variants(self) -> list[str] | None:
        """Load previously generated variants from disk cache."""
        cache_file = self._cache_path()
        if not cache_file.exists():
            return None

        try:
            with open(cache_file) as fh:
                cached = json.load(fh)
            variants = cached.get("variants", [])
            if len(variants) >= self._n_variants:
                logger.info("Loaded %d cached semantic variants from %s", len(variants), cache_file)
                return variants[:self._n_variants]
        except (json.JSONDecodeError, KeyError, OSError) as read_error:
            logger.warning("Failed to read variant cache %s: %s", cache_file, read_error)

        return None

    def _save_variants_to_cache(self, variants: list[str]) -> None:
        """Persist generated variants to disk for reuse."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_path()

        payload = {
            "seed_input": self._seed_input,
            "n_variants": self._n_variants,
            "variants": variants,
        }

        with open(cache_file, "w") as fh:
            json.dump(payload, fh, indent=2)

        logger.info("Cached %d semantic variants to %s", len(variants), cache_file)

    def _generate_variants(self) -> list[str]:
        """Generate phrasing variants using the configured provider.

        Each variant is a rephrasing of the seed input that preserves
        semantic intent but changes surface-level wording.
        """
        if self._provider is None:
            raise ValueError(
                "SemanticVariantSampler requires a provider to generate variants. "
                "Pass a configured provider instance, or use FixedSetSampler for "
                "zero-cost deterministic sampling."
            )

        generation_prompt = (
            f"Generate {self._n_variants} different phrasings of the following input. "
            f"Each phrasing must preserve the original meaning and intent but use "
            f"different words, sentence structure, or phrasing style. "
            f"Return each phrasing on its own line, numbered 1 through {self._n_variants}. "
            f"Do not include any other text.\n\n"
            f"Original input: {self._seed_input}"
        )

        response = self._provider.call(generation_prompt)
        raw_lines = response.content.strip().split("\n")

        # Parse numbered lines, stripping the number prefix
        variants = []
        for line in raw_lines:
            cleaned = line.strip()
            if not cleaned:
                continue
            # Strip common numbering patterns: "1.", "1)", "1:"
            for prefix_end in [".", ")", ":"]:
                parts = cleaned.split(prefix_end, 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    cleaned = parts[1].strip()
                    break
            if cleaned:
                variants.append(cleaned)

        return variants[:self._n_variants]

    def sample(self, n: int, seed_manager: SeedManager | None = None) -> list[InputItem]:
        # Try loading from cache first
        variants = self._load_cached_variants()

        if variants is None:
            variants = self._generate_variants()
            self._save_variants_to_cache(variants)

        if not variants:
            raise ValueError(
                "SemanticVariantSampler produced no variants. "
                "Check that the provider returned parseable numbered lines."
            )

        # Cycle through variants to fill exactly n samples
        cycled = list(itertools.islice(itertools.cycle(variants), n))
        return [InputItem(prompt=text) for text in cycled]
