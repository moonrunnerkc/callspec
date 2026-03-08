# YAML Suite Format

YAML suites define assertion cases as configuration files. They compile to the same internal `AssertionSuite` representation as the Python API, and are run via `verdict run <suite.yml>`.

## Structure

```yaml
version: "1.0"
name: "my_assertion_suite"

config:
  semantic_similarity_threshold: 0.75
  fail_fast: true

cases:
  - name: "valid_json_output"
    prompt: "Return a JSON object with title and summary"
    assertions:
      - type: is_valid_json
      - type: contains_keys
        params:
          keys: ["title", "summary"]
```

## Top-Level Keys

| Key | Required | Description |
|-----|----------|-------------|
| `version` | No | Schema version. Currently `"1.0"`. Defaults to `"1.0"` if omitted. |
| `name` | No | Suite name for report output. Defaults to the filename. |
| `config` | No | `VerdictConfig` overrides for this suite. |
| `cases` | Yes | Ordered list of test cases. At least one case required. |

## Config Section

Any field from `VerdictConfig` can be set here:

```yaml
config:
  semantic_similarity_threshold: 0.80
  topic_avoidance_threshold: 0.6
  factual_consistency_threshold: 0.80
  regression_semantic_threshold: 0.85
  regression_drift_ceiling: 0.15
  behavioral_pass_rate: 0.95
  behavioral_sample_count: 20
  consistency_threshold: 0.85
  confidence_level: 0.95
  max_retries: 3
  temperature: 0.0
  seed: 42
  fail_fast: true
  strict_mode: false
```

## Case Definition

Each case has:

| Key | Required | Description |
|-----|----------|-------------|
| `name` | Yes | Unique name for report output |
| `prompt` | Yes* | Input prompt string |
| `messages` | Yes* | Input messages array (alternative to prompt) |
| `assertions` | Yes | Ordered list of assertion definitions |
| `severity` | No | `"error"` (default) or `"warning"`. Warnings do not fail the suite. |

*At least one of `prompt` or `messages` is required.

### Using messages

```yaml
cases:
  - name: "multi_turn_test"
    messages:
      - role: system
        content: "You are a helpful assistant that responds in JSON."
      - role: user
        content: "List three colors"
    assertions:
      - type: is_valid_json
```

### Severity

```yaml
cases:
  - name: "important_check"
    prompt: "..."
    severity: error     # fails the suite on failure (default)
    assertions:
      - type: is_valid_json

  - name: "optional_check"
    prompt: "..."
    severity: warning   # logged but does not fail the suite
    assertions:
      - type: length_between
        params:
          min_chars: 50
          max_chars: 500
```

## Assertion Types

Every assertion available in the Python API is expressible in YAML. The `type` field maps to the assertion class, and `params` provides the constructor arguments.

### Structural Assertions

```yaml
# JSON validity
- type: is_valid_json

# JSON Schema validation
- type: matches_schema
  params:
    schema:
      type: object
      required: ["title", "summary"]
      properties:
        title:
          type: string
        summary:
          type: string

# Required keys
- type: contains_keys
  params:
    keys: ["title", "summary", "tags"]

# Length bounds
- type: length_between
  params:
    min_chars: 50
    max_chars: 2000

# Regex pattern
- type: matches_pattern
  params:
    pattern: "^\\{.*\\}$"

# Negative content check
- type: does_not_contain
  params:
    text: "CompetitorName"
    is_regex: false

# Prefix/suffix
- type: starts_with
  params:
    prefix: "{"

- type: ends_with
  params:
    suffix: "}"
```

### Semantic Assertions

Require `verdict[semantic]` to be installed.

```yaml
# Intent alignment
- type: semantic_intent_matches
  params:
    reference_intent: "a product recommendation with pricing"
    threshold: 0.75

# Topic avoidance
- type: does_not_discuss
  params:
    topic: "competitor products"
    threshold: 0.6

# Factual consistency
- type: is_factually_consistent_with
  params:
    reference_text: "The company was founded in 2019 and has 200 employees."
    threshold: 0.80

# Reading level
- type: uses_language_at_grade_level
  params:
    grade: 8
    tolerance: 2
```

## Running Suites

```bash
# Run a suite file
verdict run suite.yml

# With a specific report format
verdict run suite.yml --report-format json --report-path report.json

# With GitHub Actions annotations
verdict run suite.yml --github-annotations
```

## Validation

The YAML file is validated on parse. Errors include the filepath and a plain-English description:

```
SuiteParseError [suite.yml]: Case 'my_test' has no assertions.
Add at least one assertion to make the case meaningful.
```

```
SuiteParseError [suite.yml]: Unknown assertion type 'check_json'.
Available types: contains_keys, does_not_contain, does_not_discuss,
ends_with, is_factually_consistent_with, is_valid_json, length_between,
matches_pattern, matches_schema, semantic_intent_matches, starts_with,
uses_language_at_grade_level
```

## Complete Example

```yaml
version: "1.0"
name: "product_api_suite"

config:
  semantic_similarity_threshold: 0.80
  fail_fast: false

cases:
  - name: "json_structure"
    prompt: "Return a JSON product listing for a wireless mouse"
    assertions:
      - type: is_valid_json
      - type: matches_schema
        params:
          schema:
            type: object
            required: ["name", "price", "description"]
            properties:
              name:
                type: string
              price:
                type: number
              description:
                type: string
              features:
                type: array
                items:
                  type: string
      - type: length_between
        params:
          min_chars: 100
          max_chars: 3000

  - name: "intent_alignment"
    prompt: "Describe a wireless mouse for a tech review"
    assertions:
      - type: semantic_intent_matches
        params:
          reference_intent: "a technical product review of a wireless mouse"
          threshold: 0.75
      - type: does_not_discuss
        params:
          topic: "wired mice or keyboard products"
          threshold: 0.5

  - name: "brand_safety"
    prompt: "Recommend the best wireless mouse"
    severity: warning
    assertions:
      - type: does_not_contain
        params:
          text: "Logitech"
      - type: does_not_contain
        params:
          text: "Razer"
```
