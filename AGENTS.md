# AGENTS.md

## Purpose

This repository is a small, reproducible GPT-style language-model lab built
from scratch in PyTorch. Keep changes aligned with that scope: inspectable
corpus preparation, character tokenization, causal Transformer modeling,
baselines, training artifacts, run provenance, controlled generation, and
experiment reports.

Optimize for:

1. Correctness
2. Reproducibility
3. Contract clarity
4. Validation evidence
5. Maintainability
6. Honest experiment interpretation
7. Style consistency

Do not add infrastructure, scale, or abstractions unless they answer a concrete
experiment or reliability need.

## Source Of Truth

Before non-trivial work, inspect the nearest relevant files:

1. this `AGENTS.md`
2. `docs/codex/`
3. `README.md`
4. `docs/architecture.md`, `docs/training.md`, and `docs/experiments.md`
5. `pyproject.toml`, configs, scripts, tests, and touched package modules
6. latest relevant report under `experiments/`

Runtime artifacts under `data/raw/`, `data/processed/`, `checkpoints/`, and
`runs/` are local and ignored. Do not treat them as durable source unless the
task explicitly asks for an experiment run or artifact inspection.

## Repository Map

- `src/smallm/data/`: corpus normalization, stats, manifests, tokenizer, and
  token block dataset.
- `src/smallm/model/`: GPT config, causal self-attention, Transformer blocks,
  and language-model head.
- `src/smallm/evaluation/`: uniform, unigram, and bigram baselines.
- `src/smallm/training/`: trainer, artifacts, checkpoints, logging, and run
  discovery.
- `src/smallm/generation/`: text generation and sampling controls.
- `scripts/`: command-line entry points; keep these thin.
- `configs/`: smoke and GPTiny experiment configs.
- `tests/`: focused contract tests.
- `docs/`: human-facing docs.
- `docs/codex/`: agent-facing repository contracts.
- `experiments/`: chronological milestone reports.

## Operating Rules

- Keep model architecture changes separate from experiment/report-only work.
- Preserve public CLI behavior unless the task explicitly approves a change.
- Keep filesystem writes in scripts, training orchestration, or artifact helpers.
- Keep model code focused on tensors, logits, and loss.
- Do not hand-edit generated or ignored runtime artifacts unless the task is an
  explicit experiment run.
- Do not commit or rely on local corpora, checkpoints, processed data, or run
  directories.
- When changing corpus handling, preserve source metadata, checksums,
  normalization rules, and manifest compatibility.
- When changing generation, preserve controlled comparison behavior: greedy,
  seed, temperature, top-k, and max token settings must stay explicit.
- When changing training artifacts, preserve run inspectability: config,
  metrics, summary, checkpoint, sample, and dataset manifest.
- Experiment reports must be candid; negative model-quality results are valid
  outcomes when validation supports them.

## Validation

Use `docs/codex/build-and-test.md` as the validation source of truth.

Minimum expectations:

- Documentation-only changes: check links/paths where applicable.
- Python code changes: run `python -m pytest` and `python -m compileall src scripts`.
- Pipeline changes: run the closest `prepare_corpus`, `prepare_data`,
  `evaluate_baselines`, `train`, `show_run`, and `generate` commands that cover
  the changed path.
- Experiment milestones: record exact commands and observed outputs in
  `experiments/<number>-<slug>.md`.

If a relevant check cannot be run, state why and what remains unverified.

## Current Technical Status

Milestone 024 is the latest modeling evidence. On a deterministically extracted, near-size-matched
Peter Pan corpus, ByteBPE512 reaches best BPC `2.1539` versus `2.1721` for character. This
replicates the direction found on Alice, but the 0.83% margin from one seed is not a stable
effect-size estimate. Milestone 023 remains the seed-robustness reference (`2.0225 ± 0.0124` on
Alice); the next strong test is a corpus-by-seed matrix.

## Safety And Security

Treat these as review-critical:

- corpus ingestion and file paths
- network fetches for public-domain corpora
- checkpoint loading
- subprocess or shell examples in docs
- generated artifacts that may contain local data

Do not introduce remote code execution, untrusted pickle loading beyond the
existing local checkpoint workflow, or network access in tests without explicit
need and documentation.
