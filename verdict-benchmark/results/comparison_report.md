# Verdict Benchmark Results

| Metric | Verdict | DeepEval | Promptfoo | LangSmith | Braintrust |
|--------|---------|----------|-----------|-----------|------------|
| Account required | No | Partial | No | Yes | Yes |
| Language | Python | Python | Node.js | Python | Python |
| Install time (base) | 3.1s | 12.4s | -- | N/A* | N/A* |
| Install time (full semantic) | 85.3s | 12.4s | -- | N/A* | N/A* |
| First test execution | 102ms | 1491ms | -- | N/A* | N/A* |
| Total setup-to-first-test (base) | 3.2s | 13.8s | -- | N/A* | N/A* |
| Catches 0.15 drift | Yes (native) | Manual only | Manual only | Manual only | Manual only |
| Flakiness stdev (semantic) | 0.0 | 0.015972 | -- | -- | -- |
| Flakiness score range | 0.0 | 0.071429 | -- | -- | -- |
| Exit code on failure (CLI) | 1 | -- | -- | -- | -- |
| Exit code on failure (pytest) | 2 | -- | -- | -- | -- |
| Measured provider API calls | 3 | 12 | -- | -- | -- |
| Telemetry calls (uninvited) | 0 | 4 | -- | -- | -- |
| Lines of code (drift test) | 9 | 25 | 6 | 27 | -- |
| API calls per run (estimated) | 7 | 28 | 17 | 21 | 21 |
| Monthly CI cost (30/day) | $26.78 | $31.31 | $28.94 | $29.8 | $29.8 |

## GPT-4o Version Drift (gpt-4o-2024-05-13 vs gpt-4o-2024-11-20)

| Prompt | Cosine Drift | Detected (>0.15) |
|--------|-------------|-----------------|
| structured_output | 0.0121 | No |
| semantic_intent | 0.0661 | No |
| format_compliance | 0.2695 | Yes |
| code_generation | 0.1625 | Yes |
| numeric_reasoning | 0.0276 | No |
| instruction_following | 0.0370 | No |
| chain_of_thought | 0.0899 | No |

## Anthropic Migration Drift (claude-3-haiku vs claude-sonnet-4)

| Prompt | Cosine Drift | Detected (>0.15) |
|--------|-------------|-----------------|
| structured_output | 0.0995 | No |
| semantic_intent | 0.2604 | Yes |
| format_compliance | 0.4642 | Yes |
| code_generation | 0.1235 | No |
| numeric_reasoning | 0.0573 | No |
| instruction_following | 0.1211 | No |
| chain_of_thought | 0.2611 | Yes |

*LangSmith and Braintrust require account creation, which cannot be automated.
