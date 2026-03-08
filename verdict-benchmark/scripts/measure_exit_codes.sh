#!/usr/bin/env bash
# Measure CI exit code behavior on assertion failure.
#
# Tests three invocation modes per tool:
#   1. Tool's own CLI with a failing assertion
#   2. pytest with a failing assertion
#   3. Subprocess from a shell script
#
# Usage:
#   bash scripts/measure_exit_codes.sh verdict
#   bash scripts/measure_exit_codes.sh deepeval
#   bash scripts/measure_exit_codes.sh promptfoo

set -uo pipefail

TOOL="${1:?Usage: measure_exit_codes.sh <verdict|deepeval|promptfoo>}"
BENCHMARK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$BENCHMARK_ROOT/results"
mkdir -p "$RESULTS_DIR"

# Create a temporary failing suite for Verdict
create_verdict_failing_suite() {
    local tmpfile
    tmpfile=$(mktemp --suffix=.yml)
    cat > "$tmpfile" << 'YAML'
version: "1.0"
name: "failing_test"
cases:
  - name: "intentional_failure"
    prompt: "Say hello"
    assertions:
      - type: matches_pattern
        params:
          pattern: "THIS_PATTERN_WILL_NEVER_MATCH_ANY_OUTPUT_12345"
YAML
    echo "$tmpfile"
}

measure_verdict_exit_codes() {
    echo "=== Verdict Exit Codes ==="

    local suite_file
    suite_file=$(create_verdict_failing_suite)

    # Mode 1: Verdict CLI with failing assertion
    echo -n "  CLI (verdict run): "
    verdict run "$suite_file" -p mock > /dev/null 2>&1
    local cli_exit=$?
    echo "exit code $cli_exit"

    # Mode 2: pytest with failing assertion
    local pytest_file
    pytest_file=$(mktemp --suffix=.py)
    cat > "$pytest_file" << 'PYTHON'
from verdict.providers.mock import MockProvider
from verdict.verdict import Verdict

def test_intentional_failure():
    provider = MockProvider(response_fn=lambda prompt, messages=None: "Hello there!")
    v = Verdict(provider)
    result = v.assert_that("Say hello").matches_pattern("NEVER_MATCH_12345").run()
    assert result.passed, f"Assertion failed: {result.assertions[0].message}"
PYTHON
    echo -n "  pytest: "
    python -m pytest "$pytest_file" -q --tb=no --no-header > /dev/null 2>&1
    local pytest_exit=$?
    echo "exit code $pytest_exit"

    # Mode 3: subprocess
    echo -n "  subprocess: "
    python -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '$pytest_file', '-q', '--tb=no', '--no-header'],
    capture_output=True,
)
sys.exit(result.returncode)
" > /dev/null 2>&1
    local subprocess_exit=$?
    echo "exit code $subprocess_exit"

    rm -f "$suite_file" "$pytest_file"

    # Write results
    python3 -c "
import json
results_path = '$RESULTS_DIR/exit_codes.json'
try:
    with open(results_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}
data['verdict'] = {
    'cli': $cli_exit,
    'pytest': $pytest_exit,
    'subprocess': $subprocess_exit,
    'expected': {'cli': 1, 'pytest': 1, 'subprocess': 1},
    'all_nonzero': all(c != 0 for c in [$cli_exit, $pytest_exit, $subprocess_exit]),
}
with open(results_path, 'w') as f:
    json.dump(data, f, indent=2)
print(json.dumps(data['verdict'], indent=2))
"
}

measure_deepeval_exit_codes() {
    echo "=== DeepEval Exit Codes ==="

    # Mode 1: deepeval test run (requires deepeval installed)
    local pytest_file
    pytest_file=$(mktemp --suffix=.py)
    cat > "$pytest_file" << 'PYTHON'
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import assert_test

def test_intentional_failure():
    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="I enjoy eating pizza on Tuesdays.",
    )
    metric = AnswerRelevancyMetric(threshold=0.9)
    assert_test(test_case, [metric])
PYTHON

    echo -n "  deepeval test run: "
    deepeval test run "$pytest_file" > /dev/null 2>&1
    local deepeval_exit=$?
    echo "exit code $deepeval_exit"

    echo -n "  pytest: "
    python -m pytest "$pytest_file" -q --tb=no --no-header > /dev/null 2>&1
    local pytest_exit=$?
    echo "exit code $pytest_exit"

    echo -n "  subprocess: "
    python -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, '-m', 'pytest', '$pytest_file', '-q', '--tb=no', '--no-header'],
    capture_output=True,
)
sys.exit(result.returncode)
" > /dev/null 2>&1
    local subprocess_exit=$?
    echo "exit code $subprocess_exit"

    rm -f "$pytest_file"

    python3 -c "
import json
results_path = '$RESULTS_DIR/exit_codes.json'
try:
    with open(results_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}
data['deepeval'] = {
    'deepeval_cli': $deepeval_exit,
    'pytest': $pytest_exit,
    'subprocess': $subprocess_exit,
    'all_nonzero': all(c != 0 for c in [$deepeval_exit, $pytest_exit, $subprocess_exit]),
}
with open(results_path, 'w') as f:
    json.dump(data, f, indent=2)
print(json.dumps(data['deepeval'], indent=2))
"
}

measure_promptfoo_exit_codes() {
    echo "=== Promptfoo Exit Codes ==="

    # Create a failing Promptfoo config
    local config_file
    config_file=$(mktemp --suffix=.yaml)
    cat > "$config_file" << 'YAML'
providers:
  - id: echo
    config:
      text: "Hello world"
prompts:
  - "Test prompt"
tests:
  - assert:
      - type: contains
        value: "THIS_WILL_NEVER_MATCH_XYZZY_12345"
YAML

    echo -n "  npx promptfoo eval: "
    npx promptfoo eval --config "$config_file" --no-cache > /dev/null 2>&1
    local promptfoo_exit=$?
    echo "exit code $promptfoo_exit"

    rm -f "$config_file"

    python3 -c "
import json
results_path = '$RESULTS_DIR/exit_codes.json'
try:
    with open(results_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}
data['promptfoo'] = {
    'cli': $promptfoo_exit,
    'note': 'Promptfoo is Node.js only, no pytest mode available',
}
with open(results_path, 'w') as f:
    json.dump(data, f, indent=2)
print(json.dumps(data['promptfoo'], indent=2))
"
}

# -- Main dispatch --
case "$TOOL" in
    verdict)  measure_verdict_exit_codes ;;
    deepeval) measure_deepeval_exit_codes ;;
    promptfoo) measure_promptfoo_exit_codes ;;
    *) echo "Unknown tool: $TOOL"; exit 1 ;;
esac
