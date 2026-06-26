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

The next technical experiment should test capacity or tokenization before
adding unrelated infrastructure.
