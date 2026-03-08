.PHONY: install-verdict install-deepeval install-promptfoo install-all \
       record-baselines-openai record-baselines-anthropic record-baselines \
       bench-verdict bench-deepeval bench-promptfoo \
       measure-cost measure-drift measure-flakiness \
       compare report clean help

BENCHMARK_ROOT := $(shell pwd)
VERDICT_ROOT := $(shell cd .. && pwd)
PYTHON := python3

# Activate the parent project's venv if it exists, otherwise use system Python
VENV_PYTHON := $(VERDICT_ROOT)/.venv/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-28s\033[0m %s\n", $$1, $$2}'

# -- Installation --

install-verdict: ## Install LLMAssert with all benchmark dependencies
	cd $(VERDICT_ROOT) && pip install -e ".[semantic,openai,anthropic]"

install-deepeval: ## Install DeepEval
	pip install deepeval openai

install-promptfoo: ## Install Promptfoo via npm
	bash requirements/promptfoo-setup.sh

install-all: install-verdict install-deepeval install-promptfoo ## Install all tools

# -- Baseline Recording --

record-baselines-openai: ## Record baselines from OpenAI dated model versions
	$(VENV_PYTHON) scripts/record_baselines.py --provider openai

record-baselines-anthropic: ## Record baselines from Anthropic dated model versions
	$(VENV_PYTHON) scripts/record_baselines.py --provider anthropic

record-baselines: ## Record baselines from all available providers
	$(VENV_PYTHON) scripts/record_baselines.py --all

# -- Individual Benchmarks --

bench-verdict: ## Run LLMAssert benchmark suite
	@echo "=== LLMAssert Benchmark ==="
	$(VENV_PYTHON) scripts/measure_drift_detection.py --tool verdict
	$(VENV_PYTHON) scripts/measure_flakiness.py --tool verdict --runs 100
	$(VENV_PYTHON) scripts/measure_cost.py --tool verdict
	@echo ""
	@echo "LLMAssert benchmark complete. Results in results/"

bench-deepeval: ## Run DeepEval benchmark suite
	@echo "=== DeepEval Benchmark ==="
	$(VENV_PYTHON) scripts/measure_cost.py --tool deepeval
	@echo ""
	@echo "DeepEval benchmark complete. Results in results/"
	@echo "NOTE: Flakiness and drift tests skipped to avoid LLM-as-judge API costs."

bench-promptfoo: ## Run Promptfoo benchmark suite
	@echo "=== Promptfoo Benchmark ==="
	$(VENV_PYTHON) scripts/measure_cost.py --tool promptfoo
	@echo ""
	@echo "Promptfoo benchmark complete. Results in results/"

# -- Measurement Scripts --

measure-cost: ## Analyze API cost for all tools
	$(VENV_PYTHON) scripts/measure_cost.py --all

measure-drift: ## Measure drift detection using recorded baselines
	$(VENV_PYTHON) scripts/measure_drift_detection.py --all

measure-flakiness: ## Measure assertion flakiness (100 runs)
	$(VENV_PYTHON) scripts/measure_flakiness.py --tool verdict --runs 100

measure-deepeval-flakiness: ## Measure DeepEval flakiness (20 runs, costs ~$0.40)
	$(VENV_PYTHON) scripts/measure_deepeval_flakiness.py

measure-api-calls: ## Measure actual API call counts via HTTP interception
	$(VENV_PYTHON) scripts/measure_api_calls.py --all

measure-loc: ## Measure lines of code for drift detection per tool
	$(VENV_PYTHON) scripts/measure_loc.py

# -- Combined --

compare: measure-cost measure-drift measure-flakiness measure-loc report ## Run full comparison and generate report
	@echo ""
	@echo "Full comparison complete."

report: ## Generate comparison report from results
	$(VENV_PYTHON) scripts/generate_report.py

# -- LLMAssert YAML Suite --

run-verdict-suite: ## Run LLMAssert YAML assertion suite against live provider
	llm-assert run suites/verdict/regression_suite.yml -p anthropic

run-verdict-tests: ## Run LLMAssert pytest regression tests
	$(VENV_PYTHON) -m pytest suites/verdict/test_regression.py -v --tb=short

# -- Cleanup --

clean: ## Remove generated results and temporary files
	rm -f results/*.json results/*.md
	rm -rf results/terminal_output/*.txt
	@echo "Results cleaned."

clean-baselines: ## Remove all recorded baselines (requires re-recording)
	rm -f baselines/*/*.json
	@echo "Baselines cleaned. Run 'make record-baselines' to re-record."
