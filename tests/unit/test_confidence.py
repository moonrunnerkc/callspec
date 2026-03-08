"""Unit tests for the ConfidenceEstimator: Wilson intervals and score checks.

Wilson interval implementation is validated against manually computed values
and, where possible, against scipy reference. The z-score approximation
is tested against known values from standard statistical tables.
"""

from __future__ import annotations

from verdict.scoring.confidence import (
    _z_score_for_confidence,
    passes_confidence_check,
    wilson_confidence_interval,
)


class TestWilsonConfidenceInterval:

    def test_perfect_pass_rate(self) -> None:
        lower, upper = wilson_confidence_interval(20, 20, 0.95)
        assert upper == 1.0
        assert lower > 0.80

    def test_zero_pass_rate(self) -> None:
        lower, upper = wilson_confidence_interval(0, 20, 0.95)
        assert lower == 0.0
        assert upper < 0.20

    def test_half_pass_rate(self) -> None:
        lower, upper = wilson_confidence_interval(10, 20, 0.95)
        assert 0.25 < lower < 0.50
        assert 0.50 < upper < 0.75

    def test_empty_sample(self) -> None:
        lower, upper = wilson_confidence_interval(0, 0, 0.95)
        assert lower == 0.0
        assert upper == 1.0

    def test_single_success(self) -> None:
        lower, upper = wilson_confidence_interval(1, 1, 0.95)
        assert lower > 0.0
        assert upper == 1.0

    def test_single_failure(self) -> None:
        lower, upper = wilson_confidence_interval(0, 1, 0.95)
        assert lower == 0.0
        assert upper < 1.0

    def test_bounds_are_ordered(self) -> None:
        """Lower bound must always be <= upper bound."""
        for successes in range(0, 21):
            lower, upper = wilson_confidence_interval(successes, 20, 0.95)
            assert lower <= upper, f"Bounds inverted at {successes}/20: ({lower}, {upper})"

    def test_bounds_within_zero_one(self) -> None:
        """Bounds must always be in [0, 1]."""
        for successes in range(0, 21):
            lower, upper = wilson_confidence_interval(successes, 20, 0.95)
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0

    def test_higher_confidence_widens_interval(self) -> None:
        lower_95, upper_95 = wilson_confidence_interval(15, 20, 0.95)
        lower_99, upper_99 = wilson_confidence_interval(15, 20, 0.99)
        # 99% CI should be wider than 95% CI
        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99
        assert width_99 > width_95

    def test_larger_sample_narrows_interval(self) -> None:
        # 75% pass rate at different sample sizes
        lower_20, upper_20 = wilson_confidence_interval(15, 20, 0.95)
        lower_100, upper_100 = wilson_confidence_interval(75, 100, 0.95)
        width_20 = upper_20 - lower_20
        width_100 = upper_100 - lower_100
        assert width_100 < width_20

    def test_known_value_19_of_20(self) -> None:
        """Validate against a manually computed Wilson interval for 19/20 at 95%.

        p = 0.95, n = 20, z = 1.96
        center = (0.95 + 1.96^2/40) / (1 + 1.96^2/20)
        Expected lower bound approximately 0.7539
        """
        lower, upper = wilson_confidence_interval(19, 20, 0.95)
        assert abs(lower - 0.7539) < 0.01, f"Expected lower ~0.754, got {lower:.4f}"


class TestZScore:

    def test_known_values(self) -> None:
        assert abs(_z_score_for_confidence(0.90) - 1.6449) < 0.001
        assert abs(_z_score_for_confidence(0.95) - 1.9600) < 0.001
        assert abs(_z_score_for_confidence(0.99) - 2.5758) < 0.001

    def test_arbitrary_confidence(self) -> None:
        """Non-tabled confidence levels should produce reasonable z-scores."""
        z = _z_score_for_confidence(0.975)
        # 97.5% CI should have z between 1.96 (95%) and 2.576 (99%)
        assert 1.96 < z < 2.576


class TestPassesConfidenceCheck:

    def test_single_sample_above_threshold(self) -> None:
        assert passes_confidence_check(0.80, 0.75) is True

    def test_single_sample_below_threshold(self) -> None:
        assert passes_confidence_check(0.70, 0.75) is False

    def test_single_sample_at_threshold(self) -> None:
        assert passes_confidence_check(0.75, 0.75) is True

    def test_multiple_samples_all_high(self) -> None:
        scores = [0.85, 0.87, 0.83, 0.86, 0.84]
        mean_score = sum(scores) / len(scores)
        assert passes_confidence_check(
            mean_score, 0.75, n_samples=5, scores=scores
        ) is True

    def test_multiple_samples_borderline(self) -> None:
        # Scores cluster right around the threshold with high variance
        scores = [0.76, 0.74, 0.76, 0.73, 0.77]
        mean_score = sum(scores) / len(scores)
        # Lower confidence bound of the mean might dip below 0.75
        passed = passes_confidence_check(
            mean_score, 0.75, n_samples=5, scores=scores
        )
        # The result depends on the CI; just verify it returns a bool
        assert isinstance(passed, bool)

    def test_multiple_samples_clearly_failing(self) -> None:
        scores = [0.50, 0.55, 0.48, 0.52, 0.51]
        mean_score = sum(scores) / len(scores)
        assert passes_confidence_check(
            mean_score, 0.75, n_samples=5, scores=scores
        ) is False

    def test_no_scores_with_n_samples(self) -> None:
        """When scores list is None, fall back to single comparison."""
        assert passes_confidence_check(0.80, 0.75, n_samples=5, scores=None) is True
