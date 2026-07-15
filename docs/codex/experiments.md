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

Milestone 020 matched BPE character context and tested `1e-3` versus `5e-4`. Neither the matched
42-token context nor the lower rate beat the 64-token `1e-3` BPE control; future work should alter
tokenizer design rather than continue fine tuning around BPE128.

Milestone 021 altered tokenizer design with boundary-aware ByteBPE320/512. Best full-validation BPC
improved to `2.0286` and `2.0083`; the 512-token model overfit sharply after step 1,750, so the next
controlled question is early stopping or modest regularization rather than more scale.

Milestone 022 adds patience-3 validation early stopping. It reproduces ByteBPE512's step-1,750 best
checkpoint and stops at step 2,500, roughly halving runtime. Weight decay `0.01` is effectively
neutral at best BPC `2.0080`; future work should test robustness across seeds or corpus splits.

Milestone 023 runs the unregularized ByteBPE512 early-stopping setup at seeds 1337, 2027, and 4242.
Best BPC is `2.0225 ± 0.0124` (population SD), range `2.0083–2.0384`; every seed beats the
character control. The next modeling question should test split/corpus robustness, not select a seed.

Milestone 024 uses a deterministically extracted, near-size-matched Peter Pan corpus. ByteBPE512
reaches best BPC `2.1539` versus `2.1721` for character. The direction replicates, but the 0.0181
BPC margin from one shared seed is weaker evidence than a corpus-by-seed matrix.

Milestone 025 completes the balanced corpus-by-seed matrix. ByteBPE512 beats character in all six
same-seed pairs. Mean candidate-minus-reference BPC is `-0.0619` on Alice and `-0.0252` on Peter
Pan; the corpus interaction is `+0.0367`, so report a robust direction but corpus-dependent size.
The next contract change should introduce a sealed test segment before further model selection.

Milestone 026 introduces an optional `validation_split`; explicit 80/10/10 configs keep the final
region unavailable to tokenizer fitting, training, early stopping, and checkpoint selection.
One-shot best-checkpoint test evaluation confirms ByteBPE512 by `0.0614` BPC on Alice and `0.0258`
on Peter Pan. Those test segments are consumed and must not guide further tuning.

Milestone 027 was preregistered in commit `6f130d5` before Hamlet corpus access. Under the unchanged
seed-1337 sealed protocol, ByteBPE512 beats character by `0.0673` test BPC on the dramatic-play
distribution. Hamlet's terminal segment is easier than validation for both models, so gap direction
and effect magnitude remain corpus-dependent. The Hamlet test is consumed.

Milestone 028 was preregistered in commit `5ee2274` before Art of War or Lincoln corpus access.
Across three seeds per tokenizer and corpus, ByteBPE512 wins all six sealed-test pairs: mean paired
candidate-minus-character BPC is `-0.1134` on Art of War and `-0.1543` on Lincoln. Every new test
gap is positive, and the mean corpus interaction is `-0.0409`; report a robust direction with
corpus-dependent magnitude. Both terminal segments are consumed.

Milestone 029 was preregistered in commit `9de60ae` before Frankenstein, Douglass, or Origin source
access. Milestone 030 completes its 3-corpus × 3-arm × 3-seed panel. ByteBPE512 beats near-matched
char136 in 8/9 sealed pairs; mean candidate-minus-control BPC is `-0.1447`, `-0.0187`, and `-0.1717`
respectively. Douglass seed 4242 reverses by `+0.0018`. The directional hypothesis succeeds, the
strong all-nine outcome fails, every terminal segment is consumed, and version 1.0.0 closes the
project permanently.
