PYTHON ?= python

.PHONY: install format format-check lint typecheck test compile check audit smoke generate-smoke links

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

generate-smoke:
	$(PYTHON) scripts/generate.py --run latest --run-name smoke --prompt "Once" --greedy --max-new-tokens 100

links:
	$(PYTHON) scripts/check_markdown_links.py
