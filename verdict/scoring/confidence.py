"""ConfidenceEstimator: statistical bounds for probabilistic scores.

Uses Wilson score confidence intervals for proportional data (behavioral
pass rates). Wilson intervals are more accurate than normal approximation
at the extremes of the probability range, which is exactly where behavioral
assertions operate (high pass rates near 1.0 and refusal rates near 0 or 1).

Reference: scipy.stats.proportion_confint with method='wilson'.
"""

from __future__ import annotations

import math
from typing import Tuple


def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion.

    Args:
        successes: Number of successes (passing assertions).
        total: Total number of trials.
        confidence_level: Confidence level (0.95 = 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound) for the true proportion.

    The Wilson interval is preferred over the normal approximation because
    it remains accurate when the proportion is near 0 or 1, which is the
    common case for behavioral assertions (e.g., 19/20 pass rate).
    """
    if total == 0:
        return (0.0, 1.0)

    # z-score for the confidence level (two-tailed)
    z = _z_score_for_confidence(confidence_level)
    z_squared = z * z

    proportion = successes / total
    denominator = 1 + z_squared / total

    center = (proportion + z_squared / (2 * total)) / denominator
    spread = (z / denominator) * math.sqrt(
        (proportion * (1 - proportion) / total) + (z_squared / (4 * total * total))
    )

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)

    return (lower, upper)


def _z_score_for_confidence(confidence_level: float) -> float:
    """Approximate the z-score for common confidence levels.

    Uses the rational approximation of the inverse normal CDF.
    For the confidence levels used in practice (0.90, 0.95, 0.99),
    this is accurate to 4+ decimal places.
    """
    # Standard z-scores for common levels, avoiding scipy dependency
    # for this single function
    z_table = {
        0.90: 1.6449,
        0.95: 1.9600,
        0.99: 2.5758,
    }

    if confidence_level in z_table:
        return z_table[confidence_level]

    # Rational approximation (Abramowitz and Stegun, formula 26.2.23)
    # for arbitrary confidence levels
    alpha = 1.0 - confidence_level
    p = alpha / 2.0

    # Approximation of inverse normal CDF
    t = math.sqrt(-2.0 * math.log(p))
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    return z


def passes_confidence_check(
    score: float,
    threshold: float,
    confidence_level: float = 0.95,
    n_samples: int = 1,
    scores: list | None = None,
) -> bool:
    """Determine if a score passes with statistical confidence.

    For single samples (n_samples=1), this is a direct comparison.
    For multiple samples, uses the lower bound of the confidence interval
    on the mean score to determine pass/fail.
    """
    if n_samples == 1 or scores is None:
        return score >= threshold

    # For multiple samples, check that the lower confidence bound exceeds threshold
    mean_score = sum(scores) / len(scores)
    if len(scores) < 2:
        return mean_score >= threshold

    # Standard error of the mean
    variance = sum((s - mean_score) ** 2 for s in scores) / (len(scores) - 1)
    std_error = math.sqrt(variance / len(scores))
    z = _z_score_for_confidence(confidence_level)

    lower_bound = mean_score - z * std_error
    return lower_bound >= threshold
