#!/usr/bin/env bash
# Measure time from package install to first passing test.
# Reports both cold cache (fresh venv) and warm cache (cached packages) timings.
#
# Usage:
#   bash scripts/measure_setup_time.sh verdict
#   bash scripts/measure_setup_time.sh deepeval
#   bash scripts/measure_setup_time.sh promptfoo

set -euo pipefail

TOOL="${1:?Usage: measure_setup_time.sh <verdict|deepeval|promptfoo>}"
BENCHMARK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$BENCHMARK_ROOT/results"
mkdir -p "$RESULTS_DIR"

# LLMAssert has three install tiers:
#   base:          pip install verdict (jsonschema, pyyaml, click, rich)
#   with_provider: pip install "llm-assert[anthropic]" (adds anthropic SDK)
#   full:          pip install "llm-assert[anthropic,semantic]" (adds sentence-transformers, scipy, PyTorch)

VERDICT_TEST_SNIPPET='
from llm_assert.providers.mock import MockProvider
from llm_assert.verdict import LLMAssert

provider = MockProvider(response_fn=lambda prompt, messages=None: '"'"'{"status": "ok"}'"'"')
v = LLMAssert(provider)
result = v.assert_that("test").is_valid_json().run()
assert result.passed, "First assertion did not pass"
print("PASS: First LLMAssert assertion passed")
'

measure_verdict_tier() {
    local tier="$1"
    local pip_extras="$2"
    local cache_flag="$3"  # "--no-cache-dir" for cold, "" for warm
    echo "=== LLMAssert ($tier): $([ -n "$cache_flag" ] && echo 'Cold' || echo 'Warm') Cache ==="

    local venv_dir
    venv_dir=$(mktemp -d)/verdict-${tier}
    local install_start install_end install_ms
    local test_start test_end test_ms

    python -m venv "$venv_dir"

    install_start=$(date +%s%N)
    if [ -n "$pip_extras" ]; then
        "$venv_dir/bin/pip" install --quiet $cache_flag \
            -e "${BENCHMARK_ROOT}/..[$pip_extras]" 2>/dev/null
    else
        "$venv_dir/bin/pip" install --quiet $cache_flag \
            -e "$BENCHMARK_ROOT/.." 2>/dev/null
    fi
    install_end=$(date +%s%N)
    install_ms=$(( (install_end - install_start) / 1000000 ))

    test_start=$(date +%s%N)
    "$venv_dir/bin/python" -c "$VERDICT_TEST_SNIPPET"
    test_end=$(date +%s%N)
    test_ms=$(( (test_end - test_start) / 1000000 ))

    local total_ms=$(( install_ms + test_ms ))
    echo "Install: ${install_ms}ms | Test: ${test_ms}ms | Total: ${total_ms}ms"
    rm -rf "$venv_dir"
    echo "${install_ms}|${test_ms}|${total_ms}"
}

DEEPEVAL_TEST_SNIPPET='
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_case = LLMTestCase(
    input="What is 2+2?",
    actual_output="4",
)
metric = AnswerRelevancyMetric(threshold=0.5)
print("PASS: DeepEval initialized and metric created")
'

measure_deepeval() {
    local cache_flag="$1"  # "--no-cache-dir" for cold, "" for warm
    echo "=== DeepEval: $([ -n "$cache_flag" ] && echo 'Cold' || echo 'Warm') Cache ==="

    local venv_dir
    venv_dir=$(mktemp -d)/deepeval
    local install_start install_end install_ms
    local test_start test_end test_ms

    python -m venv "$venv_dir"

    install_start=$(date +%s%N)
    "$venv_dir/bin/pip" install --quiet $cache_flag deepeval openai 2>/dev/null
    install_end=$(date +%s%N)
    install_ms=$(( (install_end - install_start) / 1000000 ))

    test_start=$(date +%s%N)
    "$venv_dir/bin/python" -c "$DEEPEVAL_TEST_SNIPPET"
    test_end=$(date +%s%N)
    test_ms=$(( (test_end - test_start) / 1000000 ))

    local total_ms=$(( install_ms + test_ms ))
    echo "Install: ${install_ms}ms | Test: ${test_ms}ms | Total: ${total_ms}ms"
    rm -rf "$venv_dir"
    echo "${install_ms}|${test_ms}|${total_ms}"
}

measure_promptfoo_cold() {
    echo "=== Promptfoo: Cold Cache ==="
    local start end elapsed

    start=$(date +%s%N)

    # Promptfoo installs on first npx call if not cached
    npx --yes promptfoo@latest --version > /dev/null 2>&1

    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "Cold cache time: ${elapsed}ms (npx first-run)"
    echo "$elapsed"
}

measure_promptfoo_warm() {
    echo "=== Promptfoo: Warm Cache ==="
    local start end elapsed

    start=$(date +%s%N)

    npx promptfoo --version > /dev/null 2>&1

    end=$(date +%s%N)
    elapsed=$(( (end - start) / 1000000 ))
    echo "Warm cache time: ${elapsed}ms"
    echo "$elapsed"
}

# -- Main dispatch --

case "$TOOL" in
    verdict)
        # Measure three tiers with warm cache (representative of developer experience)
        BASE_RAW=$(measure_verdict_tier "base" "" "" 2>&1 | tail -1)
        PROVIDER_RAW=$(measure_verdict_tier "with_provider" "anthropic" "" 2>&1 | tail -1)
        FULL_RAW=$(measure_verdict_tier "full" "anthropic,semantic" "" 2>&1 | tail -1)

        IFS='|' read -r BASE_INSTALL BASE_TEST BASE_TOTAL <<< "$BASE_RAW"
        IFS='|' read -r PROV_INSTALL PROV_TEST PROV_TOTAL <<< "$PROVIDER_RAW"
        IFS='|' read -r FULL_INSTALL FULL_TEST FULL_TOTAL <<< "$FULL_RAW"

        python3 -c "
import json
results_path = '$RESULTS_DIR/setup_time.json'
try:
    with open(results_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}
data['verdict_base'] = {
    'pip_install': 'pip install verdict',
    'install_ms': int('$BASE_INSTALL'),
    'first_test_ms': int('$BASE_TEST'),
    'warm_ms': int('$BASE_TOTAL'),
}
data['verdict_with_provider'] = {
    'pip_install': 'pip install \"llm-assert[anthropic]\"',
    'install_ms': int('$PROV_INSTALL'),
    'first_test_ms': int('$PROV_TEST'),
    'warm_ms': int('$PROV_TOTAL'),
}
data['verdict_full'] = {
    'pip_install': 'pip install \"verdict[anthropic,semantic]\"',
    'install_ms': int('$FULL_INSTALL'),
    'first_test_ms': int('$FULL_TEST'),
    'warm_ms': int('$FULL_TOTAL'),
}
with open(results_path, 'w') as f:
    json.dump(data, f, indent=2)
print(json.dumps({k: v for k, v in data.items() if k.startswith('llm-assert')}, indent=2))
"
        ;;
    deepeval)
        WARM_RAW=$(measure_deepeval "" 2>&1 | tail -1)
        IFS='|' read -r DE_INSTALL DE_TEST DE_TOTAL <<< "$WARM_RAW"

        python3 -c "
import json
results_path = '$RESULTS_DIR/setup_time.json'
try:
    with open(results_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}
data['deepeval'] = {
    'pip_install': 'pip install deepeval openai',
    'install_ms': int('$DE_INSTALL'),
    'first_test_ms': int('$DE_TEST'),
    'warm_ms': int('$DE_TOTAL'),
}
with open(results_path, 'w') as f:
    json.dump(data, f, indent=2)
print(json.dumps(data['deepeval'], indent=2))
"
        ;;
    promptfoo)
        COLD=$(measure_promptfoo_cold 2>&1 | tail -1)
        WARM=$(measure_promptfoo_warm 2>&1 | tail -1)
        python3 -c "
import json
results_path = '$RESULTS_DIR/setup_time.json'
try:
    with open(results_path) as f:
        data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    data = {}
data['promptfoo'] = {'cold_ms': int('$COLD'), 'warm_ms': int('$WARM')}
with open(results_path, 'w') as f:
    json.dump(data, f, indent=2)
print(json.dumps(data['promptfoo'], indent=2))
"
        ;;
    *)
        echo "Unknown tool: $TOOL"
        exit 1
        ;;
esac
