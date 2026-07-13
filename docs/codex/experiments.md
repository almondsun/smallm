# Experiment Guide

This file defines how to create and interpret experiment milestones in
`smaLLM`.

## Purpose

Experiment reports are the durable record for model and infrastructure
milestones. They should make the question, setup, result, validation evidence,
and limitations clear enough that a future session can continue without hidden
context.

## Report Location

Use:

```text
experiments/<number>-<short-slug>.md
```

Examples:

- `experiments/010-sampling-controls.md`
- `experiments/011-larger-corpus-tiny-gpt.md`

Keep numbering monotonic. Do not rewrite older reports unless the task is to
correct a factual error.

## Required Sections

Use sections appropriate to the milestone, but include these concepts:

- goal
- setup or corpus
- implementation or method when code changed
- observed results
- validation commands
- limitations or remaining questions

For model-quality experiments, also include:

- corpus size and vocabulary size
- baseline losses and perplexities
- GPTiny validation loss and perplexity
- run path
- controlled generation examples when generation quality is discussed
- comparison against the relevant prior milestone

## Runtime Artifact Policy

Generated corpora, tokenizers, checkpoints, metrics, samples, and run
directories stay ignored. Reports may quote:

- run paths
- final losses
- checksum prefixes or full checksums
- relevant `summary.json` fields
- short generated samples

Do not add raw corpora, processed corpora, checkpoints, or run directories to
git.

## Corpus Provenance

When a corpus is prepared, record:

- source name
- source note when relevant
- raw/prepared character counts
- unique character count
- train/validation split
- checksum when important
- any manual extraction or cleaning step before `prepare_corpus.py`

If the corpus came from a network fetch, record the URL and whether boilerplate
or non-prose material was removed.

## Result Interpretation

Be direct. Do not inflate model quality.

Valid conclusions include:

- the model beat or trailed a baseline
- the validation distribution changed
- greedy decoding collapsed
- top-k sampling improved local texture but not coherence
- a larger corpus helped baselines more than GPTiny

Negative results are useful when the setup and validation are clear.

## Current Baseline Context

As of experiment 011:

- The 4.8k-character corpus run reached GPTiny validation loss `2.4505` and
  slightly beat bigram `2.5562`.
- The 144.5k-character public-domain corpus run reached GPTiny validation
  loss `2.5914` and trailed bigram `2.4340`.
- Greedy generation still collapses into repeated `the`.

As of experiment 013:

- The model/config/run family is now canonically named GPTiny / `gptiny`.
- The 2k-step GPTiny run reached validation loss `2.2187` and beat the
  larger-corpus bigram baseline.
- The optional 5k-step GPTiny run reached validation loss `1.8601`.
- Greedy generation still collapses into repeated high-probability words.

As of experiment 014:

- The 5k `lr=0.001` GPTiny run reached final validation loss `1.6792` and
  best validation loss `1.6501`, beating the 5k control.
- Dropout-off improved over the control but did not beat the higher learning
  rate run.
- Greedy generation still collapsed in every optimizer variant.

As of experiment 015:

- Wide and deep GPTiny variants improved validation over the high-LR control.
- The deep variant reached the best validation point at `1.4950`, but final
  validation rose after that while train loss kept falling.
- Generation diversity diagnostics improved with capacity, especially
  distinct-2, but generated prose still showed phrase-level repetition and weak
  coherence.

As of experiment 016:

- Simple BPE128 reduced validation sequence length from 14,453 character tokens
  to 9,522 BPE tokens.
- BPE128 underperformed the character control on estimated best bits per
  character (`2.4453` versus `2.1569`).
- BPE128 made some greedy text more word-like, but phrase-level repetition and
  incoherent prose remained.

As of experiment 017:

- Training saves both final and best-validation checkpoints.
- The BPE128 best checkpoint improved validation loss but not controlled
  generation or phrase repetition; the character control showed the same
  direction.
- The experiment 016 conclusion is unchanged: BPE128 underperformed the
  character control on estimated bits per character.

Milestone 019 supersedes the old metric contract. Full corrected evaluation reached best BPC
`2.0760` for character and `2.0976` for BPE128. BPE still shortened sequences and remained
narrowly worse, while its seeded samples retained competitive surface diversity.
