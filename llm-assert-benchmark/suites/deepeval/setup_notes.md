# DeepEval Setup Notes

## Account Requirement

DeepEval's `deepeval test run` command pushes results to Confident AI's
hosted platform by default. Running `deepeval login` is prompted on first use.

Core evaluation metrics (AnswerRelevancyMetric, FaithfulnessMetric, etc.)
can work without login when run via `pytest` directly, but several features
are gated behind the Confident AI platform:

- Historical result tracking
- Test run comparison
- Dashboard visualization
- Some advanced metrics

## Installation

```bash
pip install deepeval openai
```

## API Keys Required

- `OPENAI_API_KEY`: Required both for the model under test AND for
  DeepEval's LLM-as-judge metrics (they call OpenAI to judge the output)

## What DeepEval Cannot Do Natively

- **Baseline comparison**: No built-in mechanism to record a snapshot and
  compare future outputs against it. Drift detection requires manual code.
- **Embedding-based similarity**: DeepEval defaults to LLM-as-judge for
  semantic evaluation. No built-in cosine similarity scoring.
- **Confidence intervals**: No Wilson score or statistical flakiness
  protection. Thresholds are raw comparison values.

## Cost Impact

Every semantic assertion in DeepEval that uses LLM-as-judge makes an
additional API call. For a 20-assertion suite, that is 20 extra calls
to the judge model (typically gpt-4o-mini or gpt-4o), at approximately
$0.01-0.03 per call depending on output length.
