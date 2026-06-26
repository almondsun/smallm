# Experiment Index

This guide groups the experiment reports by milestone. It is a navigation aid,
not a replacement for the original reports.

## Milestone Map

| Experiment | Area | Result |
| --- | --- | --- |
| [001 Tiny GPT Smoke Test](../experiments/001-tiny-gpt-smoke-test.md) | First model path | Initial tokenizer, dataset, model, train script, generate script, and tests passed. |
| [002 Training Logger and Tiny Config](../experiments/002-training-logger-and-tiny-config.md) | Training visibility | Tiny config and progress logger produced readable loss/throughput output. |
| [003 Run Artifacts and Metrics](../experiments/003-run-artifacts-and-metrics.md) | Run records | Training began preserving config, metrics, summary, checkpoint, and sample files. |
| [004 Run Discovery](../experiments/004-run-discovery.md) | Run navigation | Added run listing, latest-run resolution, and run inspection. |
| [005 Real Corpus Tiny GPT](../experiments/005-real-corpus-tiny-gpt.md) | Real text | Tiny GPT trained on a 4,838-character prose corpus with vocab size 53. |
| [006 Baseline Evaluation](../experiments/006-baseline-evaluation.md) | Baselines | Tiny GPT validation loss `2.4505` beat add-one bigram `2.5562` on the small corpus. |
| [007 Reproducible Corpus Preparation](../experiments/007-reproducible-corpus-preparation.md) | Corpus prep | Baselines and training moved to a normalized prepared corpus path. |
| [008 Dataset Manifest and Checksums](../experiments/008-dataset-manifest-and-checksums.md) | Dataset identity | Corpus manifests began recording source metadata, checksums, counts, and normalization rules. |
| [009 Run Dataset Provenance](../experiments/009-run-dataset-provenance.md) | Run provenance | Runs copied `dataset_manifest.json` and stored dataset fields in `summary.json`. |
| [010 Sampling Controls](../experiments/010-sampling-controls.md) | Generation | Added `max_new_tokens`, `temperature`, `top_k`, `seed`, and greedy decoding. |
| [011 Larger Public-Domain Corpus Experiment](../experiments/011-larger-corpus-tiny-gpt.md) | Model quality | Larger corpus reached 144,530 prepared characters, but tiny GPT loss `2.5914` trailed bigram `2.4340`. |

## Topic Shortcuts

- Model path: [001](../experiments/001-tiny-gpt-smoke-test.md),
  [002](../experiments/002-training-logger-and-tiny-config.md),
  [005](../experiments/005-real-corpus-tiny-gpt.md).
- Run artifacts: [003](../experiments/003-run-artifacts-and-metrics.md),
  [004](../experiments/004-run-discovery.md),
  [009](../experiments/009-run-dataset-provenance.md).
- Data provenance: [007](../experiments/007-reproducible-corpus-preparation.md),
  [008](../experiments/008-dataset-manifest-and-checksums.md),
  [009](../experiments/009-run-dataset-provenance.md).
- Evaluation: [006](../experiments/006-baseline-evaluation.md),
  [011](../experiments/011-larger-corpus-tiny-gpt.md).
- Generation: [010](../experiments/010-sampling-controls.md),
  [011](../experiments/011-larger-corpus-tiny-gpt.md).

## Current Status

The infrastructure milestones are mostly complete for a small local lab:
prepared corpora, manifests, baselines, run artifacts, run discovery, controlled
generation, and experiment reports.

The latest model-quality milestone is negative but useful. More text alone did
not improve the unchanged 500-step tiny GPT. The next technical work should
study training budget, optimization, capacity, or tokenization.
