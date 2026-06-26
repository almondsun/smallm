# Architecture

`smaLLM` is organized as a small PyTorch package with scripts at the edge. The
main design rule is that reusable behavior lives under `src/smallm/`, while
scripts parse arguments and call package code.

## Model Path

```text
raw text
  -> prepared corpus, stats, manifest
  -> selected character or simple BPE tokenizer
  -> token block dataset
  -> GPT forward pass
  -> next-token loss
  -> checkpoint, metrics, summary, sample
  -> generation from checkpoint or run
```

## Module Boundaries

`data/` owns corpus normalization, manifest fields, tokenizer state, and token
block construction.

`model/` owns the GPT network: embeddings, causal self-attention, Transformer
blocks, final normalization, and language-model head.

`evaluation/` owns simple character-level reference models used to interpret
validation loss.

`training/` owns orchestration: dataloaders, optimization, evaluation intervals,
progress logging, checkpoints, run directories, metrics, summaries, and copied
dataset manifests.

`generation/` owns decoding from a trained model, including temperature, top-k,
seeding, and greedy mode.

`utils/` owns small runtime helpers such as device and seed selection.

## Data Contracts

Corpus preparation writes:

| File | Purpose |
| --- | --- |
| `data/processed/corpus.txt` | Normalized text used by tokenizer prep, baseline evaluation, and training. |
| `data/processed/corpus_stats.json` | Character counts, split counts, line counts, and frequency summaries. |
| `data/processed/corpus_manifest.json` | Source metadata, checksums, paths, counts, split settings, and normalization rules. |

Training copies the manifest into the run directory and stores selected dataset
identity fields in `summary.json`.

## Model Components

The GPT model is decoder-only:

- token embedding
- learned position embedding
- dropout
- repeated Transformer blocks
- final layer norm
- linear language-model head

Each block uses pre-norm residual attention and a GELU MLP. Attention uses a
lower-triangular causal mask so position `t` cannot attend to future tokens.

The forward pass returns logits and, when targets are provided, next-token
cross-entropy loss.

## Run Artifacts

Each training run writes:

| File | Purpose |
| --- | --- |
| `config.yaml` | Exact config snapshot. |
| `metrics.jsonl` | Step-level training and validation metrics. |
| `summary.json` | Final losses, duration, paths, parameter count, vocab size, dataset summary, and generation settings. |
| `checkpoint.pt` | Model state, model config, tokenizer state, and run metadata. |
| `sample.txt` | Text generated at the end of training. |
| `dataset_manifest.json` | Copied corpus manifest for run-local provenance. |

Run utilities resolve explicit run paths and `latest` by run name.

## Baselines

Baselines are intentionally small:

| Baseline | Meaning |
| --- | --- |
| Uniform | Every character has equal probability. |
| Unigram | Probability comes from train-set character frequencies. |
| Bigram | Add-one smoothed character-transition probabilities. |

The bigram baseline is the main non-neural reference in the experiment reports.

## Generation

Generation accepts:

- `max_new_tokens`
- `temperature`
- `top_k`
- `seed`
- `greedy`

Greedy mode is deterministic and useful for comparison. Sampling with a seed is
used for reproducible stochastic examples.

## Boundary Rule

The model should consume token IDs and produce next-token logits. Dataset
metadata, run provenance, filesystem writes, and CLI behavior should stay in
data, training, artifact, and script layers rather than leaking into model code.

## Deferred Work

The project intentionally does not yet include production tokenizer libraries,
larger model families, distributed training, checkpoint resume, dashboards, or
hosted tracking. The next technical work should refine tokenizer/training
comparisons before adding unrelated infrastructure.
