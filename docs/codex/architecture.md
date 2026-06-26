# Architecture Guide

This file defines the architecture contract for Codex work in this repository.
Use it with `AGENTS.md`, `docs/architecture.md`, and
`docs/codex/build-and-test.md`.

## Purpose

`smaLLM` is a small language-model lab. The code should keep the full path from
text to trained checkpoint inspectable:

```text
raw text
  -> prepared corpus, stats, manifest
  -> selected character or simple BPE tokenizer
  -> token blocks
  -> GPT logits and next-token loss
  -> checkpoint, metrics, summary, sample, copied manifest
  -> controlled generation
```

Architecture should make responsibilities, side effects, and experiment
contracts explicit.

## Layer Model

### Core Logic

Core logic is deterministic or tensor-oriented package code:

- `smallm.data.corpus`: text normalization, stats, checksums, manifests.
- `smallm.data.tokenizer`: tokenizer selection, artifact loading, and
  character-tokenizer state.
- `smallm.data.bpe_tokenizer`: educational BPE training, encode/decode, and
  inspectable JSON state.
- `smallm.data.dataset`: train/validation split and shifted token blocks.
- `smallm.model`: causal attention, Transformer blocks, GPT forward/loss.
- `smallm.evaluation`: baseline losses and perplexity.
- `smallm.generation`: decoding controls over model logits.

Core logic should be easy to unit test and should avoid hidden global state.

### Orchestration

Orchestration coordinates workflows:

- `smallm.training.trainer`: training loop, evaluation intervals, checkpoint
  save, sample generation, summary writing.
- `smallm.training.runs`: run discovery and latest-run resolution.
- `smallm.training.logging`: progress formatting and console output.

Orchestration may call multiple subsystems and write artifacts, but should keep
policy explicit in config and summary fields.

### Edge Scripts

Scripts under `scripts/` are CLI adapters. They should parse arguments, load
config/checkpoints, call package code, and print results. Do not put reusable
domain logic in scripts when it belongs under `src/smallm/`.

## Module Boundaries

`data/` owns:

- conservative corpus normalization
- source and split metadata
- raw/prepared checksums
- character or simple BPE tokenizer state
- token block dataset construction

`model/` owns:

- tensor shapes
- causal masking
- Transformer block behavior
- next-token logits and optional loss

`evaluation/` owns:

- baseline probability models
- validation loss and perplexity calculations

`training/` owns:

- dataloaders and optimization
- metrics and progress logging
- run directory creation
- checkpoint and summary artifacts
- dataset manifest copy and summary projection

`generation/` owns:

- greedy decoding
- temperature
- top-k filtering
- seeded stochastic sampling
- max token limits

## Boundary Rules

- The model consumes token IDs and returns logits/loss. It should not know about
  file paths, corpora, manifests, runs, or CLIs.
- Dataset provenance belongs in corpus/artifact/training code, not model code.
- Generation settings must remain explicit in CLI flags and run summaries.
- Baselines should stay independent from the neural model path.
- Config defaults may support experiments, but experiment conclusions belong in
  `experiments/`, not in code comments.
- If a change modifies a persisted artifact schema, call out compatibility
  impact and update tests/docs.

## Artifact Contracts

Corpus preparation writes:

- `data/processed/corpus.txt`
- `data/processed/corpus_stats.json`
- `data/processed/corpus_manifest.json`

Training run directories write:

- `config.yaml`
- `metrics.jsonl`
- `summary.json`
- `checkpoint.pt`
- `sample.txt`
- `dataset_manifest.json`

Do not remove or rename these artifacts without an explicit migration decision.

## Current Constraints

The project intentionally does not yet include:

- production tokenizer libraries or a broad tokenizer framework
- larger model families
- distributed training
- checkpoint resume
- mixed precision
- dashboards
- remote tracking

Before adding any of these, prefer a small experiment that identifies the
specific limitation being addressed.
