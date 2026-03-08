"""Unit tests for the sampling module: SeedManager, FixedSetSampler,
TemplateSampler, and SemanticVariantSampler.

Tests cover determinism, cycling, shuffling, template expansion,
cache behavior, and error handling across all sampler strategies.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from verdict.providers.mock import MockProvider
from verdict.sampling.sampler import InputItem
from verdict.sampling.seed import SeedManager
from verdict.sampling.strategies import (
    FixedSetSampler,
    SemanticVariantSampler,
    TemplateSampler,
)

# ---------------------------------------------------------------------------
# SeedManager
# ---------------------------------------------------------------------------


class TestSeedManager:
    """SeedManager determinism and reset behavior."""

    def test_deterministic_sequence_from_same_root(self):
        """Same root seed always produces the same sequence of seeds."""
        manager_a = SeedManager(root_seed=99)
        manager_b = SeedManager(root_seed=99)
        sequence_a = [manager_a.next_seed() for _ in range(50)]
        sequence_b = [manager_b.next_seed() for _ in range(50)]
        assert sequence_a == sequence_b

    def test_different_roots_produce_different_sequences(self):
        manager_a = SeedManager(root_seed=1)
        manager_b = SeedManager(root_seed=2)
        sequence_a = [manager_a.next_seed() for _ in range(20)]
        sequence_b = [manager_b.next_seed() for _ in range(20)]
        assert sequence_a != sequence_b

    def test_reset_replays_the_same_sequence(self):
        manager = SeedManager(root_seed=42)
        first_pass = [manager.next_seed() for _ in range(30)]
        manager.reset()
        second_pass = [manager.next_seed() for _ in range(30)]
        assert first_pass == second_pass

    def test_get_rng_returns_independent_random(self):
        """get_rng produces a Random instance that does not perturb the main sequence."""
        manager = SeedManager(root_seed=7)
        _consumed = manager.next_seed()  # advance once
        rng = manager.get_rng()                # consumes a seed internally

        # The rng should produce deterministic output from its own seed
        values = [rng.randint(0, 100) for _ in range(10)]
        assert len(set(values)) > 1  # not degenerate

        # The main sequence should not be affected by rng usage
        next_val = manager.next_seed()
        manager.reset()
        _consumed_again = manager.next_seed()
        _ = manager.get_rng()
        assert manager.next_seed() == next_val

    def test_root_seed_property(self):
        manager = SeedManager(root_seed=123)
        assert manager.root_seed == 123

    def test_none_root_seed(self):
        manager = SeedManager(root_seed=None)
        assert manager.root_seed is None
        # Should still produce seeds without crashing
        seed = manager.next_seed()
        assert isinstance(seed, int)


# ---------------------------------------------------------------------------
# InputItem
# ---------------------------------------------------------------------------


class TestInputItem:
    """InputItem is a frozen dataclass with prompt and optional messages."""

    def test_create_with_prompt_only(self):
        item = InputItem(prompt="hello")
        assert item.prompt == "hello"
        assert item.messages is None

    def test_create_with_messages(self):
        msgs = [{"role": "user", "content": "hi"}]
        item = InputItem(prompt="hello", messages=msgs)
        assert item.messages == msgs

    def test_frozen(self):
        item = InputItem(prompt="hello")
        with pytest.raises(AttributeError):
            item.prompt = "goodbye"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FixedSetSampler
# ---------------------------------------------------------------------------


class TestFixedSetSampler:
    """FixedSetSampler with various input types and cycling behavior."""

    def test_strings_normalized_to_input_items(self):
        sampler = FixedSetSampler(["a", "b", "c"])
        items = sampler.sample(3)
        assert all(isinstance(item, InputItem) for item in items)
        assert [item.prompt for item in items] == ["a", "b", "c"]

    def test_input_items_passed_through(self):
        originals = [InputItem(prompt="x"), InputItem(prompt="y")]
        sampler = FixedSetSampler(originals)
        items = sampler.sample(2)
        assert items == originals

    def test_dict_inputs_normalized(self):
        dicts = [
            {"prompt": "hello", "messages": [{"role": "user", "content": "hi"}]},
            {"prompt": "world"},
        ]
        sampler = FixedSetSampler(dicts)
        items = sampler.sample(2)
        assert items[0].prompt == "hello"
        assert items[0].messages == [{"role": "user", "content": "hi"}]
        assert items[1].prompt == "world"
        assert items[1].messages is None

    def test_cycling_when_n_exceeds_list_length(self):
        sampler = FixedSetSampler(["a", "b"])
        items = sampler.sample(5)
        assert [item.prompt for item in items] == ["a", "b", "a", "b", "a"]

    def test_exact_n_returns_exact_count(self):
        sampler = FixedSetSampler(["x", "y", "z"])
        items = sampler.sample(2)
        assert len(items) == 2
        assert [item.prompt for item in items] == ["x", "y"]

    def test_shuffle_with_seed_manager_is_deterministic(self):
        sampler = FixedSetSampler(["a", "b", "c", "d", "e"], shuffle=True)
        seed_a = SeedManager(root_seed=10)
        seed_b = SeedManager(root_seed=10)
        items_a = sampler.sample(5, seed_manager=seed_a)
        items_b = sampler.sample(5, seed_manager=seed_b)
        assert [i.prompt for i in items_a] == [i.prompt for i in items_b]

    def test_shuffle_actually_reorders(self):
        """Shuffle should produce a different order than the original at least once."""
        original = list("abcdefghij")
        sampler = FixedSetSampler(original, shuffle=True)
        seed = SeedManager(root_seed=77)
        items = sampler.sample(10, seed_manager=seed)
        shuffled_order = [i.prompt for i in items]
        # With 10 items and a random shuffle, the probability of getting
        # the original order is 1/10! which is vanishingly small
        assert shuffled_order != original

    def test_empty_inputs_raises(self):
        sampler = FixedSetSampler([])
        with pytest.raises(ValueError, match="no inputs"):
            sampler.sample(5)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="int"):
            FixedSetSampler([123])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# TemplateSampler
# ---------------------------------------------------------------------------


class TestTemplateSampler:
    """TemplateSampler expansion, validation, and cycling."""

    def test_single_slot_expansion(self):
        sampler = TemplateSampler(
            template="Hello {name}!",
            variables={"name": ["Alice", "Bob", "Charlie"]},
        )
        items = sampler.sample(3)
        prompts = [item.prompt for item in items]
        assert prompts == ["Hello Alice!", "Hello Bob!", "Hello Charlie!"]

    def test_multiple_slots_cartesian_product(self):
        sampler = TemplateSampler(
            template="{color} {animal}",
            variables={"color": ["red", "blue"], "animal": ["cat", "dog"]},
        )
        items = sampler.sample(4)
        prompts = sorted(item.prompt for item in items)
        expected = sorted(["blue cat", "blue dog", "red cat", "red dog"])
        assert prompts == expected

    def test_cycling_with_exhaustive(self):
        sampler = TemplateSampler(
            template="Q: {topic}?",
            variables={"topic": ["math", "science"]},
        )
        items = sampler.sample(5)
        prompts = [item.prompt for item in items]
        assert prompts == [
            "Q: math?", "Q: science?", "Q: math?", "Q: science?", "Q: math?"
        ]

    def test_random_mode_is_deterministic_with_seed(self):
        sampler = TemplateSampler(
            template="{x}",
            variables={"x": ["a", "b", "c", "d", "e"]},
            exhaustive=False,
        )
        seed_a = SeedManager(root_seed=55)
        seed_b = SeedManager(root_seed=55)
        items_a = sampler.sample(5, seed_manager=seed_a)
        items_b = sampler.sample(5, seed_manager=seed_b)
        assert [i.prompt for i in items_a] == [i.prompt for i in items_b]

    def test_missing_slot_raises(self):
        with pytest.raises(ValueError, match="slots without variable lists"):
            TemplateSampler(
                template="Hello {name}, welcome to {place}!",
                variables={"name": ["Alice"]},
            )

    def test_returns_input_items(self):
        sampler = TemplateSampler(
            template="test {v}",
            variables={"v": ["1"]},
        )
        items = sampler.sample(1)
        assert isinstance(items[0], InputItem)
        assert items[0].messages is None

    def test_n_less_than_total_combinations(self):
        sampler = TemplateSampler(
            template="{x}",
            variables={"x": ["a", "b", "c", "d", "e"]},
        )
        items = sampler.sample(3)
        assert len(items) == 3


# ---------------------------------------------------------------------------
# SemanticVariantSampler
# ---------------------------------------------------------------------------


class TestSemanticVariantSampler:
    """SemanticVariantSampler generation, caching, and error handling."""

    def _mock_variant_response(self, prompt, messages=None):
        """Simulates an LLM producing numbered variant lines."""
        return (
            "1. What is the weather like today?\n"
            "2. How is the weather right now?\n"
            "3. Could you tell me today's weather?\n"
            "4. What does the forecast look like today?\n"
            "5. Is it going to rain today?\n"
        )

    def test_generates_variants_from_provider(self):
        provider = MockProvider(response_fn=self._mock_variant_response)
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = SemanticVariantSampler(
                seed_input="What's the weather today?",
                n_variants=5,
                provider=provider,
                cache_dir=tmpdir,
            )
            items = sampler.sample(5)
            assert len(items) == 5
            assert all(isinstance(item, InputItem) for item in items)
            # Variants should be the cleaned text, not numbered
            assert not items[0].prompt.startswith("1.")

    def test_caches_to_disk(self):
        provider = MockProvider(response_fn=self._mock_variant_response)
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = SemanticVariantSampler(
                seed_input="What's the weather today?",
                n_variants=5,
                provider=provider,
                cache_dir=tmpdir,
            )
            sampler.sample(3)

            # Cache file should exist
            cache_files = list(Path(tmpdir).glob("*.json"))
            assert len(cache_files) == 1

            # Parse the cache file to verify structure
            with open(cache_files[0]) as fh:
                cached = json.load(fh)
            assert "variants" in cached
            assert len(cached["variants"]) == 5
            assert cached["seed_input"] == "What's the weather today?"

    def test_loads_from_cache_without_provider_call(self):
        """On second run, variants are loaded from cache, no provider needed."""
        call_count = 0

        def counting_response(prompt, messages=None):
            nonlocal call_count
            call_count += 1
            return self._mock_variant_response(prompt, messages)

        provider = MockProvider(response_fn=counting_response)
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = SemanticVariantSampler(
                seed_input="cached test",
                n_variants=5,
                provider=provider,
                cache_dir=tmpdir,
            )
            # First call generates variants
            sampler.sample(5)
            first_call_count = call_count

            # Second call should use cache
            sampler_2 = SemanticVariantSampler(
                seed_input="cached test",
                n_variants=5,
                provider=provider,
                cache_dir=tmpdir,
            )
            sampler_2.sample(5)
            assert call_count == first_call_count, "Provider should not be called on cache hit"

    def test_cycles_variants_to_fill_n(self):
        provider = MockProvider(response_fn=self._mock_variant_response)
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = SemanticVariantSampler(
                seed_input="test cycling",
                n_variants=3,
                provider=provider,
                cache_dir=tmpdir,
            )
            items = sampler.sample(7)
            assert len(items) == 7
            # First 3 and next 3 should be the same sequence
            first_three = [i.prompt for i in items[:3]]
            second_three = [i.prompt for i in items[3:6]]
            assert first_three == second_three

    def test_no_provider_raises(self):
        sampler = SemanticVariantSampler(
            seed_input="test",
            n_variants=5,
            provider=None,
        )
        with pytest.raises(ValueError, match="requires a provider"):
            sampler.sample(5)

    def test_cache_key_determinism(self):
        """Same seed input and n_variants always produce the same cache key."""
        sampler_a = SemanticVariantSampler(seed_input="hello", n_variants=10)
        sampler_b = SemanticVariantSampler(seed_input="hello", n_variants=10)
        assert sampler_a._cache_key() == sampler_b._cache_key()

    def test_different_inputs_produce_different_cache_keys(self):
        sampler_a = SemanticVariantSampler(seed_input="hello", n_variants=10)
        sampler_b = SemanticVariantSampler(seed_input="goodbye", n_variants=10)
        assert sampler_a._cache_key() != sampler_b._cache_key()

    def test_different_n_variants_produce_different_cache_keys(self):
        sampler_a = SemanticVariantSampler(seed_input="hello", n_variants=5)
        sampler_b = SemanticVariantSampler(seed_input="hello", n_variants=15)
        assert sampler_a._cache_key() != sampler_b._cache_key()

    def test_handles_provider_output_with_varied_numbering(self):
        """Parser handles different numbering formats: 1), 1:, 1."""

        def varied_response(prompt, messages=None):
            return (
                "1) First variant\n"
                "2: Second variant\n"
                "3. Third variant\n"
                "plain text variant\n"
            )

        provider = MockProvider(response_fn=varied_response)
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = SemanticVariantSampler(
                seed_input="test",
                n_variants=4,
                provider=provider,
                cache_dir=tmpdir,
            )
            items = sampler.sample(4)
            prompts = [i.prompt for i in items]
            assert "First variant" in prompts
            assert "Second variant" in prompts
            assert "Third variant" in prompts
            assert "plain text variant" in prompts

    def test_corrupt_cache_regenerates(self):
        """A corrupt cache file triggers regeneration instead of crashing."""
        provider = MockProvider(response_fn=self._mock_variant_response)
        with tempfile.TemporaryDirectory() as tmpdir:
            sampler = SemanticVariantSampler(
                seed_input="corrupt test",
                n_variants=5,
                provider=provider,
                cache_dir=tmpdir,
            )
            # Write a corrupt cache file
            cache_file = sampler._cache_path()
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as fh:
                fh.write("{invalid json!!!}")

            # Should regenerate without error
            items = sampler.sample(5)
            assert len(items) == 5
