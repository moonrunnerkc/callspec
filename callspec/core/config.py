"""Global configuration for Callspec assertion runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Calibrated against SBERT STS-B benchmarks (Reimers & Gurevych, 2019).
# 0.75 cosine similarity in the all-MiniLM-L6-v2 space corresponds to
# "clearly semantically related" per human annotator agreement.
SEMANTIC_SIMILARITY_THRESHOLD = 0.75

# Lower than the positive-match threshold by design: catches responses
# that approach a prohibited topic even loosely.
TOPIC_AVOIDANCE_THRESHOLD = 0.6

# Factual consistency requires tighter alignment with the reference.
FACTUAL_CONSISTENCY_THRESHOLD = 0.80

# Regression drift: 15% semantic distance is the default ceiling before
# a baseline comparison fails.
REGRESSION_DRIFT_CEILING = 0.15

# Regression structural + semantic combined threshold.
REGRESSION_SEMANTIC_THRESHOLD = 0.85

# Behavioral pass rate: 95% of samples must pass the inner assertion.
BEHAVIORAL_PASS_RATE = 0.95

# Default sample count for behavioral assertions. 20 is the minimum
# for meaningful binary inference; see architecture doc section 9.4.
BEHAVIORAL_SAMPLE_COUNT = 20

# Consistency threshold for is_consistent_across_samples.
CONSISTENCY_THRESHOLD = 0.85

# Wilson confidence interval level for probabilistic assertions.
CONFIDENCE_LEVEL = 0.95

# Provider retry policy: retries on network/rate-limit errors, not assertion failures.
MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SECONDS = 1.0

# Default embedding model for semantic scoring. 22MB, CPU-native, no API key.
# MTEB benchmark validates this as the best size/quality tradeoff for English STS.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class CallspecConfig:
    """Central configuration object threaded through the entire run.

    Every numeric constant with behavioral meaning is configurable here
    rather than scattered across assertion implementations. Defaults are
    calibrated and documented; override only with justification.
    """

    # Semantic scoring
    semantic_similarity_threshold: float = SEMANTIC_SIMILARITY_THRESHOLD
    topic_avoidance_threshold: float = TOPIC_AVOIDANCE_THRESHOLD
    factual_consistency_threshold: float = FACTUAL_CONSISTENCY_THRESHOLD
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    # Regression
    regression_semantic_threshold: float = REGRESSION_SEMANTIC_THRESHOLD
    regression_drift_ceiling: float = REGRESSION_DRIFT_CEILING

    # Behavioral
    behavioral_pass_rate: float = BEHAVIORAL_PASS_RATE
    behavioral_sample_count: int = BEHAVIORAL_SAMPLE_COUNT
    consistency_threshold: float = CONSISTENCY_THRESHOLD
    confidence_level: float = CONFIDENCE_LEVEL

    # Provider
    max_retries: int = MAX_RETRIES
    retry_backoff_base_seconds: float = RETRY_BACKOFF_BASE_SECONDS
    temperature: float = 0.0
    seed: int | None = 42

    # Execution
    fail_fast: bool = True
    strict_mode: bool = False

    # Extensibility: provider-specific or assertion-specific overrides
    extra: dict[str, Any] = field(default_factory=dict)
