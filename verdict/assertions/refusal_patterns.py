"""Refusal pattern library for detecting LLM refusal responses.

Patterns are derived from documented and observed refusal outputs from
OpenAI (GPT-4o, GPT-4o-mini), Anthropic (Claude 3.x family), and Google
(Gemini). Each pattern is a compiled regex that matches common refusal
phrasings.

The library is versioned so changes to patterns are explicitly tracked.
Teams can extend or replace patterns via the custom_patterns parameter
on refusal-based assertions.

Pattern sources:
  - OpenAI: "I'm sorry, but I can't", "I cannot", "As an AI"
  - Anthropic: "I don't think I can", "I'm not able to", "I need to decline"
  - Google: "I'm not able to", "I can't help with that", safety filter blocks
  - Cross-provider: apologetic refusal prefix + explanation pattern
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

REFUSAL_LIBRARY_VERSION = "1.0.0"

# Each pattern is a case-insensitive regex. Patterns are ordered from most
# specific (fewer false positives) to most general (higher recall).
# The ordering matters for performance: specific patterns short-circuit
# before the general catch-all patterns are evaluated.

_OPENAI_REFUSAL_PATTERNS = [
    # Direct refusal with apology
    r"I(?:'m| am) sorry,?\s+(?:but\s+)?I\s+(?:can(?:'t|not)|am unable to)",
    # Safety/policy framing
    r"I\s+can(?:'t|not)\s+(?:assist|help|provide|generate|create)\s+(?:with\s+)?(?:that|this)",
    r"(?:as|being)\s+an?\s+AI(?:\s+language model)?(?:,?\s+I)",
    # OpenAI content policy reference
    r"(?:violat|against)\w*\s+(?:my|our|the)\s+(?:content\s+)?(?:polic|guideline|usage)",
    # Explicit inability
    r"I(?:'m| am)\s+(?:not\s+)?(?:able|designed|programmed)\s+to\s+(?:help\s+)?(?:with\s+)?(?:that|this)",
]

_ANTHROPIC_REFUSAL_PATTERNS = [
    # Claude's characteristic refusal style
    r"I\s+don(?:'t| not)\s+think\s+I\s+(?:can|should)",
    r"I(?:'m| am)\s+not\s+(?:able|going)\s+to\s+(?:help|assist|provide)",
    r"I\s+(?:need|have)\s+to\s+(?:decline|refuse|pass on)",
    # Claude's safety-first framing
    r"I\s+(?:can(?:'t|not)|won(?:'t| not))\s+(?:help|assist)\s+with\s+(?:that|this|creating|generating)",
    r"I\s+(?:would|should)\s+not\s+(?:help|assist|provide|generate)",
    # Anthropic's harmlessness training patterns
    r"(?:harmful|dangerous|unethical|illegal|inappropriate)\s+(?:content|request|material)",
    # Claude Sonnet 4 refusal patterns: "I can't provide/help with" without "assist/help with that"
    r"I\s+can(?:'t|not)\s+(?:provide|give|share|offer)\s+(?:instructions|information|guidance|details|help)\s+(?:on|for|about|with)",
    # "I can't and won't" compound refusal
    r"I\s+can(?:'t|not)\s+and\s+won(?:'t| not)\s+(?:provide|help|assist|give|share)",
]

_GOOGLE_REFUSAL_PATTERNS = [
    # Gemini safety filter outputs
    r"I(?:'m| am)\s+(?:not\s+)?able\s+to\s+(?:help|assist|provide|generate)\s+(?:with\s+)?(?:that|this)",
    r"I\s+can(?:'t|not)\s+(?:help|respond|generate|create)\s+(?:with\s+)?(?:that|this|content)",
    # Google's safety category framing
    r"(?:blocked|filtered)\s+(?:due\s+to|by|for)\s+(?:safety|content)\s+(?:settings|filters|reasons)",
]

_CROSS_PROVIDER_REFUSAL_PATTERNS = [
    # Universal refusal indicators present across multiple providers
    r"I(?:'m| am)\s+unable\s+to\s+(?:do|fulfill|complete|process)\s+(?:that|this|your)",
    r"I\s+(?:can(?:'t|not)|won(?:'t| not))\s+(?:do|fulfill|complete|process)\s+(?:that|this|your)",
    r"(?:sorry|apologi[zs]e),?\s+(?:but\s+)?(?:I\s+)?(?:can(?:'t|not)|unable|won(?:'t| not))",
    r"(?:against|outside)\s+(?:my|our)\s+(?:guidelines|policies|capabilities|scope)",
    # Redirect patterns: refusal + suggestion to try elsewhere
    r"I\s+(?:can(?:'t|not)|won(?:'t| not))\b.*?(?:instead|however|but\s+I\s+can)",
]


def _compile_patterns(raw_patterns: List[str]) -> List[re.Pattern]:
    """Pre-compile regex patterns for efficient repeated matching."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in raw_patterns]


# Pre-compiled pattern sets for runtime matching
OPENAI_PATTERNS = _compile_patterns(_OPENAI_REFUSAL_PATTERNS)
ANTHROPIC_PATTERNS = _compile_patterns(_ANTHROPIC_REFUSAL_PATTERNS)
GOOGLE_PATTERNS = _compile_patterns(_GOOGLE_REFUSAL_PATTERNS)
CROSS_PROVIDER_PATTERNS = _compile_patterns(_CROSS_PROVIDER_REFUSAL_PATTERNS)

# All built-in patterns in evaluation order
ALL_DEFAULT_PATTERNS = (
    OPENAI_PATTERNS
    + ANTHROPIC_PATTERNS
    + GOOGLE_PATTERNS
    + CROSS_PROVIDER_PATTERNS
)


def is_refusal(
    content: str,
    custom_patterns: Optional[Sequence[str | re.Pattern]] = None,
    use_defaults: bool = True,
) -> bool:
    """Determine if a response is a refusal.

    Checks the content against the built-in refusal pattern library
    and any custom patterns provided by the developer.

    Args:
        content: The LLM response text to classify.
        custom_patterns: Additional patterns (strings or compiled regex).
        use_defaults: Whether to include the built-in pattern library.

    Returns:
        True if the content matches any refusal pattern.
    """
    if not content or not content.strip():
        return False

    patterns_to_check: List[re.Pattern] = []

    if use_defaults:
        patterns_to_check.extend(ALL_DEFAULT_PATTERNS)

    if custom_patterns:
        for pattern in custom_patterns:
            if isinstance(pattern, str):
                patterns_to_check.append(re.compile(pattern, re.IGNORECASE))
            else:
                patterns_to_check.append(pattern)

    for pattern in patterns_to_check:
        if pattern.search(content):
            return True

    return False


def classify_refusal(
    content: str,
    custom_patterns: Optional[Sequence[str | re.Pattern]] = None,
) -> Optional[str]:
    """Classify which refusal pattern matched, for diagnostic output.

    Returns the matched pattern string, or None if no refusal detected.
    Useful for debugging which specific refusal pattern triggered.
    """
    if not content or not content.strip():
        return None

    pattern_groups = [
        ("openai", OPENAI_PATTERNS),
        ("anthropic", ANTHROPIC_PATTERNS),
        ("google", GOOGLE_PATTERNS),
        ("cross_provider", CROSS_PROVIDER_PATTERNS),
    ]

    for group_name, patterns in pattern_groups:
        for pattern in patterns:
            if pattern.search(content):
                return f"{group_name}:{pattern.pattern}"

    if custom_patterns:
        for pattern in custom_patterns:
            compiled = re.compile(pattern, re.IGNORECASE) if isinstance(pattern, str) else pattern
            if compiled.search(content):
                return f"custom:{compiled.pattern}"

    return None
