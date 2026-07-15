# Experiment Guide

This file defines how to interpret the completed experiment record in `smaLLM`. The curated human
index is [`docs/experiments.md`](../experiments.md); the original reports under `experiments/` are
the durable evidence.

## Report Contract

Experiment reports use `experiments/<number>-<short-slug>.md` with monotonic numbering. Historical
reports preserve the question, setup, observed result, validation, and limitations known at that
milestone. Do not rewrite them to match later conclusions; add an explicit erratum when a factual
or metric contract was superseded.

For model-quality reports, record:

- corpus identity, size, split, and checksum;
- tokenizer and vocabulary information;
- model configuration, seed, budget, and run path;
- relevant baselines and character-normalized BPC for cross-tokenizer comparisons;
- exact validation commands and observed outputs;
- controlled generation settings when discussing generated text;
- limitations, negative results, and the applicable evidence boundary.

## Runtime Artifact Policy

Generated corpora, tokenizers, checkpoints, metrics, samples, and run directories stay ignored.
Reports may quote bounded fields, checksums, run paths, and short samples, but must not commit raw
corpora, processed corpora, checkpoints, or local run directories.

Network-sourced corpora must record the source URL, extraction boundary, normalization, raw and
prepared checksums, character counts, and chronological split. Test regions remain unavailable to
tokenizer fitting, training, early stopping, and checkpoint selection.

## Interpretation Rules

- Report negative and mixed results directly; do not turn surface fluency into a quality claim.
- Compare tokenizers with exact character-normalized BPC, not token loss or perplexity.
- Treat validation as selection data and sealed test regions as one-shot confirmatory evidence.
- Preserve preregistered arms, seeds, exclusions, stopping rules, and contrasts after data access.
- Do not use any consumed terminal segment to select later work.
- Treat fixed corpora and seeds as a bounded panel, not a population sample.

## Final State

Milestone 030 completes the preregistered 3-corpus × 3-arm × 3-seed capacity panel and closes the
research roadmap. ByteBPE512 beats near-parameter-matched char136 in eight of nine sealed pairs,
with a narrow Douglass reversal and strongly corpus-dependent effect sizes. Every reported test
segment is consumed. Version 1.0.0 is feature completion; version 1.0.1 is archive polish only.

No new experiment should be added to this repository. Independent continuation belongs in a fork.
