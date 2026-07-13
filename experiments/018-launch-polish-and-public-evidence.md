# 018 Launch Polish and Public Evidence Package

## Goal

Make `smaLLM` easy to inspect as a public learning and research-engineering
artifact without changing the model or running a new experiment.

## Changes

- Added push and pull-request CI for pytest and compileall on Python 3.11.
- Added honest CI, Python requirement, and MIT license badges.
- Added a README reviewer path and a compact evidence table sourced from
  experiments 011–017.
- Added a Mermaid pipeline diagram covering character/BPE selection, training
  artifacts, best checkpoints, generation diagnostics, and reports.
- Added simple Make targets for installation, checks, smoke training,
  smoke-run generation, and Markdown link validation.
- Added a reusable Markdown path checker and aligned the focused training and
  validation documentation.

## What This Improves for Reviewers

A reviewer can now identify the project contract, inspect the main code
boundaries, read the strongest evidence, and reproduce repository checks from
the README and standard top-level commands. The results table exposes both
improvements and negative findings without requiring a full chronological read.

## Intentionally Not Changed

- No model architecture, tokenizer, configuration, or experiment conclusion.
- No new model training or BPE sweep.
- No dashboard, hosted tracking, external tokenizer, or package-download
  integration.
- No generated corpus, tokenizer, checkpoint, or run artifact was added to
  version control.

## Validation

Commands:

```bash
python -m pytest
python -m compileall src scripts
python scripts/check_markdown_links.py
make check
make links
```

The workflow YAML was inspected for event, Python, install, test, and compile
paths. GitHub Actions was not run locally.

Observed results:

- `python -m pytest`: 57 passed.
- `python -m compileall src scripts`: passed.
- `python scripts/check_markdown_links.py`: passed.
- `make check`: passed, including 57 tests and compileall.
- `make links`: passed.
- CI YAML parsed successfully and contained every required command and path.
- No training command was run for this milestone.

## Remaining Launch Tasks

No repository-local blocker is known. Publishing the current branch and
confirming the first hosted GitHub Actions run are external launch steps.
