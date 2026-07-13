# Build And Test Guide

This file defines the canonical validation path for Codex work in this
repository. Prefer these commands over generic guesses.

## Project Type

- Python package under `src/smallm/`.
- CLI scripts under `scripts/`.
- Tests under `tests/`.
- Build metadata in `pyproject.toml`.
- Common local commands in `Makefile`.
- GitHub Actions validation in `.github/workflows/ci.yml`.
- Runtime artifacts under ignored `data/`, `checkpoints/`, and `runs/` paths.

Ruff, mypy, pytest coverage, compileall, and Markdown links form the quality baseline.

## Installation

Frozen development install:

```bash
uv sync --frozen --extra dev
```

Editable pip fallback:

```bash
python -m pip install -e ".[dev]"
```

The package requires Python `>=3.10`, PyTorch, and NumPy. Development installs
also include PyYAML and pytest.

## Canonical Checks

Run the standard local checks together with:

```bash
make check
```

Audit the frozen environment against the current vulnerability database with `make audit`.

The individual commands remain canonical and are listed below.

### Unit Tests

```bash
python -m pytest
```

Current expected result after milestone 026+: at least 163 tests passing with at least 90% coverage.

### Compile Check

```bash
python -m compileall src scripts
```

Run this after Python code changes and before finalizing broad documentation
changes that include command examples.

### Documentation Link/Path Check

Use the checked-in Markdown path checker when links are changed:

```bash
make links
# equivalent to: python scripts/check_markdown_links.py
```

## Continuous Integration

`.github/workflows/ci.yml` runs on pushes and pull requests with Python 3.10 and 3.12.
It installs the frozen `uv` environment, then runs `make check`.

CI intentionally has no coverage service or external tracking integration.

## Pipeline Validation

Use these when changing the training/evaluation/generation flow or creating an
experiment report.

### Prepare Corpus

```bash
python scripts/prepare_corpus.py \
  --input data/raw/input.txt \
  --output data/processed/corpus.txt \
  --stats data/processed/corpus_stats.json \
  --manifest data/processed/corpus_manifest.json \
  --source-name "local text corpus"
```

Use `--source-note` for fetch URLs, extraction rules, or curation notes.

### Prepare Tokenizer

```bash
python scripts/prepare_data.py --config configs/gptiny.yaml
```

For fast smoke validation, use `configs/smoke.yaml`.

### Baselines

```bash
python scripts/evaluate_baselines.py --config configs/gptiny.yaml
```

### Training

```bash
python scripts/train.py --config configs/gptiny.yaml
```

For quick end-to-end checks:

```bash
python scripts/train.py --config configs/smoke.yaml
```

Training writes ignored artifacts under `runs/<run-name>/<run-id>/`.

### Run Inspection

```bash
python scripts/show_run.py --run latest --run-name gptiny
```

### Generation

```bash
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --greedy --max-new-tokens 100
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 100
```

### Sealed Test Evaluation

For an explicitly configured three-way split, evaluate only after the checkpoint decision is frozen:

```bash
python scripts/evaluate_test.py --run runs/<name>/<id> --checkpoint-kind best
```

Do not rerun, delete, or use the resulting test artifact for model selection.

## Validation Matrix

| Change type | Minimum validation |
| --- | --- |
| Documentation only | Link/path check for changed docs. |
| Config only | `python -m pytest`; run relevant script with the config if behavior changed. |
| Corpus prep or manifest logic | Corpus prep command, `python -m pytest`, compileall. |
| Tokenizer or dataset logic | `python -m pytest`, compileall; run `prepare_data.py` if behavior changed. |
| Model code | `python -m pytest`, compileall. |
| Baselines | `python -m pytest`, `evaluate_baselines.py`, compileall. |
| Training/artifacts/runs | smoke or tiny training path, `show_run.py`, `python -m pytest`, compileall. |
| Split or sealed-test contract | Legacy smoke plus three-way fixture, one-shot evaluator, `python -m pytest`, compileall. |
| Generation | greedy and seeded top-k generation commands, `python -m pytest`, compileall. |
| Experiment report | Commands claimed in the report, plus tests and compileall when code changed. |

## Reporting Validation

Final responses and experiment reports should distinguish:

- commands run
- pass/fail result
- observed run path when training ran
- generated outputs when generation quality is part of the task
- unverified behavior when a command was skipped

Do not claim a validation command passed unless it was actually run in the
current task or explicitly quoted from a prior report.

## Current Milestone Status

Milestone 027 uses the milestone-026 evaluation contract: legacy two-way splits remain compatible,
while explicit validation fractions reserve a chronological test region that training leaves
unencoded and unscored.
Best-checkpoint test evaluation verifies corpus/checkpoint identity, requires full coverage, and
refuses artifact overwrite. The preregistered Hamlet replication confirms the workflow on an
external dramatic-play distribution.
