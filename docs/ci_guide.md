# CI Guide

Integration recipes for running LLMAssert in continuous integration pipelines.

## GitHub Actions

### Using the LLMAssert Action

The simplest setup uses the composite action:

```yaml
name: LLM Tests
on: [push, pull_request]

jobs:
  llm-assert:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: moonrunnerkc/llm-assert@v1
        with:
          suite: tests/llm_assert_suite.yml
          llm-assert-extras: openai,semantic
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The action:
1. Sets up Python
2. Installs LLMAssert with specified extras
3. Runs the assertion suite
4. Annotates the PR with failures (inline on the diff)
5. Writes a step summary to the workflow run UI
6. Sets output variables for downstream steps

### Action Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `command` | `run` | LLMAssert command (`run`, `pytest`, `check`) |
| `suite` | `""` | Path to YAML suite file |
| `report-format` | `json` | `json`, `junit`, or `plaintext` |
| `report-path` | `llm-assert-report.json` | Output path for the report |
| `python-version` | `3.12` | Python version |
| `llm-assert-extras` | `""` | Comma-separated extras (e.g., `openai,semantic`) |
| `working-directory` | `.` | Working directory |
| `strict` | `false` | Enable strict mode |

### Action Outputs

| Output | Description |
|--------|-------------|
| `passed` | `true` or `false` |
| `passed-cases` | Number of passing cases |
| `failed-cases` | Number of failing cases |
| `total-cases` | Total number of cases |
| `report-path` | Path to the generated report |

### Using pytest Instead

Run LLMAssert as part of your existing pytest suite:

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install
        run: pip install -e ".[openai,semantic,dev]"

      - name: Run tests
        run: pytest --llm-assert-report json --llm-assert-report-path llm-assert-report.json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: llm-assert-report
          path: llm-assert-report.json
```

### PR Annotations

When running in GitHub Actions, LLMAssert can emit workflow command annotations that appear directly on the PR diff:

```
::error file=tests/test_summarizer.py,line=15,title=LLMAssert: semantic_intent_matches::
SemanticAssertion failed: score 0.6823 below threshold 0.7500 using all-MiniLM-L6-v2
```

To enable annotations programmatically:

```python
from llm_assert.integrations.github_actions import is_github_actions, emit_suite_result

if is_github_actions():
    emit_suite_result(suite_result, suite_name="my_suite", file="tests/test_llm.py")
```

### Caching the Embedding Model

Cache the sentence-transformers model to avoid downloading 22MB on every run:

```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/huggingface
    key: sentence-transformers-minilm-v2

- name: Pre-download model
  run: python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Two-Tier CI Strategy

Run structural and semantic tests on every commit. Run behavioral tests on a schedule or pre-release:

```yaml
# On every push: fast tests only
name: Fast Tests
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[openai,semantic,dev]"
      - run: pytest --llm-assert-skip-behavioral
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

```yaml
# Nightly: full behavioral suite
name: Behavioral Tests
on:
  schedule:
    - cron: "0 3 * * *"
jobs:
  behavioral:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[openai,semantic,dev]"
      - run: pytest -m llm_assert_behavioral -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## GitLab CI

```yaml
# .gitlab-ci.yml
llm-assert:
  image: python:3.12
  stage: test
  script:
    - pip install "llm-assert[openai,semantic]"
    - pip install -e .
    - llm-assert run tests/llm_assert_suite.yml --report-format junit --report-path llm-assert.xml
  artifacts:
    reports:
      junit: llm-assert.xml
  variables:
    OPENAI_API_KEY: $OPENAI_API_KEY
```

## CircleCI

```yaml
# .circleci/config.yml
version: 2.1
jobs:
  llm-assert:
    docker:
      - image: cimg/python:3.12
    steps:
      - checkout
      - run:
          name: Install
          command: pip install "llm-assert[openai,semantic]" -e .
      - run:
          name: Run LLMAssert
          command: llm-assert run tests/llm_assert_suite.yml --report-format junit --report-path test-results/llm-assert.xml
      - store_test_results:
          path: test-results

workflows:
  test:
    jobs:
      - llm-assert
```

## Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent { docker { image 'python:3.12' } }
    environment {
        OPENAI_API_KEY = credentials('openai-api-key')
    }
    stages {
        stage('Install') {
            steps {
                sh 'pip install "llm-assert[openai,semantic]" -e .'
            }
        }
        stage('Test') {
            steps {
                sh 'llm-assert run tests/llm_assert_suite.yml --report-format junit --report-path llm-assert.xml'
            }
            post {
                always {
                    junit 'llm_assert.xml'
                }
            }
        }
    }
}
```

## General CI Best Practices

**Exit codes:** `llm-assert run` exits with code 0 on success, 1 on assertion failure. CI pipelines can use this directly.

**API key management:** Store API keys in CI secrets. Never commit keys to the repository.

**Cost control:** Behavioral assertions make N provider calls per assertion. Use `--llm-assert-skip-behavioral` on every-commit runs. Reserve behavioral tests for nightly or pre-release pipelines.

**Offline testing:** Use `MockProvider` for tests that validate assertion configuration without API spend. Real provider tests run only when keys are available.

**Report archiving:** Save report files as CI artifacts for historical comparison. The JSON format is designed for ingestion by verdict.run for trend tracking.
