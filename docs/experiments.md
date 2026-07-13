# Experiment Index

This guide groups the experiment reports by milestone. It is a navigation aid,
not a replacement for the original reports.

## Milestone Map

| Experiment | Area | Result |
| --- | --- | --- |
| [001 Tiny GPT Smoke Test](../experiments/001-tiny-gpt-smoke-test.md) | First model path | Initial tokenizer, dataset, model, train script, generate script, and tests passed under the original name. |
| [002 Training Logger and Tiny Config](../experiments/002-training-logger-and-tiny-config.md) | Training visibility | Tiny config and progress logger produced readable loss/throughput output. |
| [003 Run Artifacts and Metrics](../experiments/003-run-artifacts-and-metrics.md) | Run records | Training began preserving config, metrics, summary, checkpoint, and sample files. |
| [004 Run Discovery](../experiments/004-run-discovery.md) | Run navigation | Added run listing, latest-run resolution, and run inspection. |
| [005 Real Corpus Tiny GPT](../experiments/005-real-corpus-tiny-gpt.md) | Real text | GPTiny, then named Tiny GPT, trained on a 4,838-character prose corpus with vocab size 53. |
| [006 Baseline Evaluation](../experiments/006-baseline-evaluation.md) | Baselines | GPTiny validation loss `2.4505` beat add-one bigram `2.5562` on the small corpus. |
| [007 Reproducible Corpus Preparation](../experiments/007-reproducible-corpus-preparation.md) | Corpus prep | Baselines and training moved to a normalized prepared corpus path. |
| [008 Dataset Manifest and Checksums](../experiments/008-dataset-manifest-and-checksums.md) | Dataset identity | Corpus manifests began recording source metadata, checksums, counts, and normalization rules. |
| [009 Run Dataset Provenance](../experiments/009-run-dataset-provenance.md) | Run provenance | Runs copied `dataset_manifest.json` and stored dataset fields in `summary.json`. |
| [010 Sampling Controls](../experiments/010-sampling-controls.md) | Generation | Added `max_new_tokens`, `temperature`, `top_k`, `seed`, and greedy decoding. |
| [011 Larger Public-Domain Corpus Experiment](../experiments/011-larger-corpus-tiny-gpt.md) | Model quality | Larger corpus reached 144,530 prepared characters, but GPTiny loss `2.5914` trailed bigram `2.4340`. |
| [012 Documentation and Portfolio Narrative](../experiments/012-documentation-and-portfolio-narrative.md) | Documentation | Reframed the project docs around the current reproducible language-model lab. |
| [013 GPTiny Training Budget and Optimization](../experiments/013-gptiny-training-budget-and-optimization.md) | Training budget | Renamed the model family to GPTiny and found 2k/5k-step runs beat the larger-corpus bigram baseline. |
| [014 Optimizer and Sampling Diagnostics](../experiments/014-optimizer-and-sampling-diagnostics.md) | Optimizer diagnostics | A 5k `lr=0.001` run beat the 5k control, but greedy generation still collapsed. |
| [015 GPTiny Capacity and Generation Diagnostics](../experiments/015-gptiny-capacity-and-generation-diagnostics.md) | Capacity diagnostics | Wider/deeper GPTiny improved validation and generation diversity metrics, but prose remained incoherent. |
| [016 Tokenization Study](../experiments/016-tokenization-study.md) | Tokenization | Simple BPE128 shortened sequences and made some greedy text more word-like, but underperformed the character control on estimated bits per character. |
| [017 Best-Checkpoint Evaluation](../experiments/017-best-checkpoint-evaluation.md) | Checkpoint diagnostics | Added best-validation checkpoints; best checkpoints improved validation but not controlled generation for BPE128 or the character control. |
| [018 Launch Polish and Public Evidence](../experiments/018-launch-polish-and-public-evidence.md) | Launch polish | Added reviewer navigation, CI, local checks, and a public evidence summary. |
| [019 Professionalization and Corrected Evaluation](../experiments/019-professionalization-and-corrected-evaluation.md) | Scientific hardening | Corrects tokenizer leakage and validation coverage, hardens artifacts and quality gates, adds the theory handbook, and reruns the controlled comparison. |
| [020 BPE Context and Learning Rate](../experiments/020-bpe-context-and-learning-rate.md) | Tokenizer diagnostics | Matching character context and lowering BPE learning rate did not beat the corrected BPE128 control or character model. |

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
  [011](../experiments/011-larger-corpus-tiny-gpt.md),
  [016](../experiments/016-tokenization-study.md).
- Generation: [010](../experiments/010-sampling-controls.md),
  [011](../experiments/011-larger-corpus-tiny-gpt.md),
  [013](../experiments/013-gptiny-training-budget-and-optimization.md),
  [014](../experiments/014-optimizer-and-sampling-diagnostics.md),
  [015](../experiments/015-gptiny-capacity-and-generation-diagnostics.md),
  [016](../experiments/016-tokenization-study.md),
  [017](../experiments/017-best-checkpoint-evaluation.md).
- Tokenization: [016](../experiments/016-tokenization-study.md).

## Current Status

The infrastructure milestones are mostly complete for a small local lab:
prepared corpora, manifests, baselines, run artifacts, run discovery, controlled
generation, and experiment reports.

Experiment 017 added best-validation checkpoint saving and a controlled
final-versus-best generation comparison. The best checkpoint improved BPE128
validation loss but not generated text or phrase repetition; the character
control showed the same direction. This does not change experiment 016's
conclusion that BPE128 underperformed the character control on estimated bits
per character.

Milestone 019 corrects the old evaluation contract. Full held-out evaluation reached best BPC
`2.0760` for character and `2.0976` for BPE128. The BPE gap is much smaller than previously
reported, but it did not beat the character control.
