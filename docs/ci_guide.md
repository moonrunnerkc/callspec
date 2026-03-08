# CI Guide

Integration recipes for running Verdict in continuous integration pipelines.

## GitHub Actions

### Using the Verdict Action

The simplest setup uses the composite action:

```yaml
name: LLM Tests
on: [push, pull_request]

jobs:
  verdict:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: moonrunnerkc/verdict@v1
        with:
          suite: tests/verdict_suite.yml
          verdict-extras: openai,semantic
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The action:
1. Sets up Python
2. Installs Verdict with specified extras
3. Runs the assertion suite
4. Annotates the PR with failures (inline on the diff)
5. Writes a step summary to the workflow run UI
6. Sets output variables for downstream steps

### Action Inputs

| Input | Default | Description |
|-------|---------|-------------|
| `command` | `run` | Verdict command (`run`, `pytest`, `check`) |
| `suite` | `""` | Path to YAML suite file |
| `report-format` | `json` | `json`, `junit`, or `plaintext` |
| `report-path` | `verdict-report.json` | Output path for the report |
| `python-version` | `3.12` | Python version |
| `verdict-extras` | `""` | Comma-separated extras (e.g., `openai,semantic`) |
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

Run Verdict as part of your existing pytest suite:

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
        run: pytest --verdict-report json --verdict-report-path verdict-report.json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: verdict-report
          path: verdict-report.json
```

### PR Annotations

When running in GitHub Actions, Verdict can emit workflow command annotations that appear directly on the PR diff:

```
::error file=tests/test_summarizer.py,line=15,title=Verdict: semantic_intent_matches::
SemanticAssertion failed: score 0.6823 below threshold 0.7500 using all-MiniLM-L6-v2
```

To enable annotations programmatically:

```python
from verdict.integrations.github_actions import is_github_actions, emit_suite_result

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
      - run: pytest --verdict-skip-behavioral
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
      - run: pytest -m verdict_behavioral -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## GitLab CI

```yaml
# .gitlab-ci.yml
verdict:
  image: python:3.12
  stage: test
  script:
    - pip install "verdict[openai,semantic]"
    - pip install -e .
    - verdict run tests/verdict_suite.yml --report-format junit --report-path verdict.xml
  artifacts:
    reports:
      junit: verdict.xml
  variables:
    OPENAI_API_KEY: $OPENAI_API_KEY
```

## CircleCI

```yaml
# .circleci/config.yml
version: 2.1
jobs:
  verdict:
    docker:
      - image: cimg/python:3.12
    steps:
      - checkout
      - run:
          name: Install
          command: pip install "verdict[openai,semantic]" -e .
      - run:
          name: Run Verdict
          command: verdict run tests/verdict_suite.yml --report-format junit --report-path test-results/verdict.xml
      - store_test_results:
          path: test-results

workflows:
  test:
    jobs:
      - verdict
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
                sh 'pip install "verdict[openai,semantic]" -e .'
            }
        }
        stage('Test') {
            steps {
                sh 'verdict run tests/verdict_suite.yml --report-format junit --report-path verdict.xml'
            }
            post {
                always {
                    junit 'verdict.xml'
                }
            }
        }
    }
}
```

## General CI Best Practices

**Exit codes:** `verdict run` exits with code 0 on success, 1 on assertion failure. CI pipelines can use this directly.

**API key management:** Store API keys in CI secrets. Never commit keys to the repository.

**Cost control:** Behavioral assertions make N provider calls per assertion. Use `--verdict-skip-behavioral` on every-commit runs. Reserve behavioral tests for nightly or pre-release pipelines.

**Offline testing:** Use `MockProvider` for tests that validate assertion configuration without API spend. Real provider tests run only when keys are available.

**Report archiving:** Save report files as CI artifacts for historical comparison. The JSON format is designed for ingestion by verdict.run for trend tracking.
