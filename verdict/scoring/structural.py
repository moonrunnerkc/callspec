"""StructuralScorer: schema validation, regex, and format checks.

Thin wrapper around jsonschema and regex operations used by structural assertions.
Exists as a scorer module so structural scoring logic is centralized rather than
duplicated across individual assertion classes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


def validate_json(content: str) -> Tuple[bool, Optional[str], Optional[int]]:
    """Attempt to parse content as JSON.

    Returns (success, error_message, error_position).
    """
    try:
        json.loads(content)
        return (True, None, None)
    except json.JSONDecodeError as parse_error:
        return (False, parse_error.msg, parse_error.pos)


def validate_schema(content: str, schema: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    """Validate parsed JSON against a JSON Schema.

    Returns (valid, list_of_violations). Each violation is a dict with
    'path', 'message', and 'validator' keys.
    """
    import jsonschema

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return (False, [{"path": [], "message": "Content is not valid JSON", "validator": "json"}])

    validator = jsonschema.Draft7Validator(schema)
    violations = [
        {
            "path": list(error.absolute_path),
            "message": error.message,
            "validator": error.validator,
        }
        for error in validator.iter_errors(parsed)
    ]

    return (len(violations) == 0, violations)


def compute_flesch_kincaid_grade(text: str) -> float:
    """Compute Flesch-Kincaid grade level for English text.

    Formula validated continuously since 1948. Purely arithmetic,
    deterministic, and transparent. Returns a float representing
    the US grade level required to understand the text.

    Returns 0.0 for empty or unparseable text rather than raising.
    """
    sentences = _count_sentences(text)
    words = _count_words(text)
    syllables = _count_syllables(text)

    if words == 0 or sentences == 0:
        return 0.0

    # Flesch-Kincaid Grade Level formula
    grade = (
        0.39 * (words / sentences)
        + 11.8 * (syllables / words)
        - 15.59
    )

    return max(0.0, grade)


def _count_sentences(text: str) -> int:
    """Count sentences by splitting on terminal punctuation."""
    # Split on period, exclamation, question mark followed by space or end
    sentences = re.split(r'[.!?]+', text)
    # Filter empty strings from splitting
    return max(1, len([s for s in sentences if s.strip()]))


def _count_words(text: str) -> int:
    """Count words by splitting on whitespace."""
    return len(text.split())


def _count_syllables(text: str) -> int:
    """Approximate syllable count using vowel-group heuristic.

    Not perfect for all English words, but sufficient for grade-level
    estimation across typical LLM output lengths.
    """
    text = text.lower()
    words = text.split()
    total = 0

    for word in words:
        # Strip non-alpha characters
        word = re.sub(r'[^a-z]', '', word)
        if not word:
            continue

        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in 'aeiouy'
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Silent 'e' at end
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        # Every word has at least one syllable
        total += max(1, syllable_count)

    return total
