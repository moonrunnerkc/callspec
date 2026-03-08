"""EmbeddingScorer: cosine similarity via sentence-transformers.

Uses all-MiniLM-L6-v2 by default (22MB, CPU-native, no API key).
The model is loaded lazily on first use and cached for the process lifetime.
Cosine similarity is used over dot product because it normalizes for vector
magnitude, producing stable comparisons between short and long responses.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Lazy-loaded singleton to avoid importing sentence-transformers at module level.
# This keeps `import llm_assert` fast when semantic features are not used.
_model_cache: dict = {}


def _get_model(model_name: str):
    """Load and cache the sentence-transformers model.

    Fails loudly with actionable instructions if the model cannot be loaded,
    rather than silently falling back or producing a cryptic import error.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as import_error:
        raise ImportError(
            "Semantic assertions require sentence-transformers. "
            "Install with: pip install llm-assert[semantic]"
        ) from import_error

    try:
        model = SentenceTransformer(model_name)
    except Exception as load_error:
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. "
            f"If running in CI, ensure the model is pre-downloaded and cached. "
            f"Error: {load_error}"
        ) from load_error

    _model_cache[model_name] = model
    logger.info("Loaded embedding model: %s", model_name)
    return model


def compute_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    """Encode a list of texts into embedding vectors.

    Returns a 2D numpy array of shape (len(texts), embedding_dim).
    The model is loaded lazily and cached.
    """

    model = _get_model(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Returns a float in [-1, 1]. Identical vectors produce 1.0.
    Normalizes for magnitude so comparisons between short and long
    texts are stable.
    """
    import numpy as np  # noqa: F811 -- lazy import, numpy ships with sentence-transformers

    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    # Guard against zero-norm vectors (empty string embeddings)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def score_similarity(text_a: str, text_b: str, model_name: str) -> float:
    """Compute semantic similarity between two texts.

    Convenience function that handles embedding and similarity in one call.
    Returns a float in [-1, 1].
    """
    embeddings = compute_embeddings([text_a, text_b], model_name)
    return cosine_similarity(embeddings[0], embeddings[1])
