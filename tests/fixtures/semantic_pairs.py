"""Semantic test fixtures: known-good and known-bad text pairs.

Each pair is documented with the expected similarity range and the reasoning
for why it should pass or fail at the default thresholds. Reviewers and
contributors can inspect these to understand threshold calibration.

These fixtures are grounded in the all-MiniLM-L6-v2 embedding space.
Changing the embedding model may shift similarity scores and require
recalibration.
"""

# -- Intent matching pairs --
# Default threshold: 0.75 (SBERT STS-B "clearly semantically related")

INTENT_MATCH_GOOD_PAIRS = [
    {
        "content": "The effects of climate change on coastal areas include rising sea levels, "
                   "coastal erosion, and economic damage to waterfront properties.",
        "intent": "the effects of climate change on coastal areas",
        "reason": "Direct semantic alignment between content and intent. "
                  "Empirically measured similarity ~0.89 in all-MiniLM-L6-v2.",
    },
    {
        "content": "To install and set up the llm-assert package, run pip install llm-assert "
                   "in your terminal.",
        "intent": "how to install and set up the llm-assert package",
        "reason": "Content directly addresses the installation and setup intent. "
                  "Empirically measured similarity ~0.86.",
    },
    {
        "content": "The company reported strong financial performance and growth, "
                   "with revenue increasing 15% year-over-year.",
        "intent": "the company reported strong financial performance and growth",
        "reason": "Revenue and growth vocabulary shared between content and intent. "
                  "Empirically measured similarity ~0.86.",
    },
]

INTENT_MATCH_BAD_PAIRS = [
    {
        "content": "The recipe calls for two cups of flour, one egg, and a pinch of salt.",
        "intent": "explain quantum computing principles",
        "reason": "Cooking recipe vs quantum computing. No semantic overlap. "
                  "Expected similarity < 0.3.",
    },
    {
        "content": "Our new smartwatch features a 1.5-inch AMOLED display "
                   "with 5ATM water resistance.",
        "intent": "describe wildlife conservation efforts in Africa",
        "reason": "Consumer electronics vs wildlife conservation. Unrelated domains.",
    },
]

# -- Topic avoidance pairs --
# Default threshold: 0.6 (lower, catches loose proximity)

TOPIC_AVOIDANCE_GOOD_PAIRS = [
    {
        "content": "Our software helps teams manage their sprint planning "
                   "and track development velocity across projects.",
        "topic": "competitor products and pricing",
        "reason": "Content about project management, does not mention competitors. "
                  "Expected similarity < 0.6.",
    },
]

TOPIC_AVOIDANCE_BAD_PAIRS = [
    {
        "content": "While our product is great, CompetitorX offers similar features "
                   "at a lower price point which some users prefer.",
        "topic": "competitor products and pricing",
        "reason": "Directly discusses competitor pricing. "
                  "Expected similarity >= 0.6.",
    },
]

# -- Factual consistency pairs --
# Default threshold: 0.80 (tighter alignment with reference)

FACTUAL_CONSISTENCY_GOOD_PAIRS = [
    {
        "content": "The study found that regular exercise reduces the risk of "
                   "heart disease by approximately 30%, according to the 2024 report.",
        "reference": "A 2024 medical study demonstrated that consistent physical activity "
                     "lowers cardiovascular disease risk by roughly 30%.",
        "reason": "Same claim, different phrasing. Strong semantic consistency.",
    },
]

FACTUAL_CONSISTENCY_BAD_PAIRS = [
    {
        "content": "The ancient Egyptians built the Great Wall of China "
                   "using advanced alien technology.",
        "reference": "The Great Wall of China was built over many centuries by "
                     "various Chinese dynasties, beginning in the 7th century BC.",
        "reason": "Contradicts the reference on who built it and how. "
                  "Expected low consistency score.",
    },
]

# -- Grade level test texts --

GRADE_LEVEL_TEXTS = [
    {
        "text": "The cat sat on the mat. It was a good cat. The mat was red.",
        "expected_range": (0, 4),
        "reason": "Very simple sentences, short words. Grade 1-3 range.",
    },
    {
        "text": (
            "The implementation of distributed consensus algorithms in Byzantine "
            "fault-tolerant systems requires careful consideration of network partition "
            "scenarios, message ordering guarantees, and the fundamental impossibility "
            "results established by Fischer, Lynch, and Paterson in their seminal 1985 paper."
        ),
        "expected_range": (14, 30),
        "reason": "Academic prose with long sentences and polysyllabic technical terms. "
                  "Graduate level or above.",
    },
]
