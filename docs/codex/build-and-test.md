# Build And Test Guide

This file defines the canonical validation path for Codex work in this
repository. Prefer these commands over generic guesses.

## Project Type

- Python package under `src/smallm/`.
- CLI scripts under `scripts/`.
- Tests under `tests/`.
- Build metadata in `pyproject.toml`.
- Runtime artifacts under ignored `data/`, `checkpoints/`, and `runs/` paths.

There is no Makefile, CI workflow, type-checker config, linter config, or
formatter config currently checked in.

## Installation

Editable install with development dependencies:

```bash
python -m pip install -e ".[dev]"
```

The package requires Python `>=3.10`, PyTorch, and NumPy. Development installs
also include PyYAML and pytest.

## Canonical Checks

### Unit Tests

```bash
python -m pytest
```

Current expected result after milestone 016+: 54 tests passing.

### Compile Check

```bash
python -m compileall src scripts
```

Run this after Python code changes and before finalizing broad documentation
changes that include command examples.

### Documentation Link/Path Check

There is no checked-in docs linter. For documentation-only changes, use a small
Markdown path check when links are changed:

```bash
python - <<'PY'
from pathlib import Path
import re

files = [Path("README.md"), *Path("docs").glob("*.md"), *Path("docs/codex").glob("*.md"), Path("experiments/README.md")]
missing = []
for file in files:
    text = file.read_text(encoding="utf-8")
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = match.group(1)
        if "://" in target or target.startswith("#"):
            continue
        path = (file.parent / target.split("#", 1)[0]).resolve()
        if not path.exists():
            missing.append((str(file), target))
if missing:
    for item in missing:
        print("missing:", item)
    raise SystemExit(1)
print("markdown links resolve")
PY
```

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
