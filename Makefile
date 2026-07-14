PYTHON ?= python

.PHONY: install format format-check lint typecheck test compile check audit smoke demo generate-smoke links chart

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest --cov=smallm --cov-report=term-missing

format:
	$(PYTHON) -m ruff format src scripts tests

format-check:
	$(PYTHON) -m ruff format --check src scripts tests

lint:
	$(PYTHON) -m ruff check src scripts tests

typecheck:
	$(PYTHON) -m mypy

compile:
	$(PYTHON) -m compileall src scripts

check: format-check lint typecheck test compile links

audit:
	$(PYTHON) -m pip_audit

smoke:
	$(PYTHON) scripts/train.py --config configs/smoke.yaml

demo:
	$(PYTHON) scripts/prepare_corpus.py --input examples/demo_corpus.txt --output data/processed/demo_corpus.txt --stats data/processed/demo_corpus_stats.json --manifest data/processed/demo_corpus_manifest.json --source-name "smaLLM synthetic demo corpus"
	$(PYTHON) scripts/prepare_data.py --config configs/demo.yaml
	$(PYTHON) scripts/evaluate_baselines.py --config configs/demo.yaml
	$(PYTHON) scripts/train.py --config configs/demo.yaml
	$(PYTHON) scripts/show_run.py --run latest --run-name demo
	$(PYTHON) scripts/generate.py --run latest --run-name demo --prompt "Research" --greedy --max-new-tokens 40

generate-smoke:
	$(PYTHON) scripts/generate.py --run latest --run-name smoke --prompt "Once" --greedy --max-new-tokens 100

links:
	$(PYTHON) scripts/check_markdown_links.py

chart:
	$(PYTHON) scripts/render_results_chart.py
