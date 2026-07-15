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
| [021 Boundary-Aware Byte BPE](../experiments/021-boundary-aware-byte-bpe.md) | Tokenizer design | Lossless boundary-aware ByteBPE320/512 beat both corrected controls on best BPC; ByteBPE512 reached `2.0083` but overfit early. |
| [022 Early Stopping and Regularization](../experiments/022-early-stopping-and-regularization.md) | Training control | Patience-3 stopping reproduced the step-1750 optimum and halved runtime; weight decay `0.01` was effectively neutral. |
| [023 Multi-Seed Robustness](../experiments/023-multi-seed-robustness.md) | Robustness | Three preregistered seeds average best BPC `2.0225 ± 0.0124`; every seed beats the corrected character control. |
| [024 Cross-Corpus Robustness](../experiments/024-cross-corpus-robustness.md) | External validity | On near-size-matched Peter Pan, ByteBPE512 narrowly beats character at `2.1539` versus `2.1721` BPC. |
| [025 Corpus-by-Seed Matrix](../experiments/025-corpus-by-seed-matrix.md) | Factorial robustness | ByteBPE512 wins all six paired comparisons; mean advantage is `0.0619` BPC on Alice and `0.0252` on Peter Pan. |
| [026 Sealed Test Evaluation](../experiments/026-sealed-test-evaluation.md) | Confirmatory evaluation | On untouched terminal segments, ByteBPE512 beats character by `0.0614` BPC on Alice and `0.0258` on Peter Pan. |
| [027 Hamlet External-Distribution Replication](../experiments/027-hamlet-external-distribution.md) | Preregistered external validity | ByteBPE512 beats character by `0.0673` sealed-test BPC on a dramatic play; the terminal region is easier than validation for both models. |
| [028 Preregistered External Corpus Panel](../experiments/028-preregistered-external-corpus-panel.md) | Multi-corpus confirmatory panel | ByteBPE512 wins all six same-seed sealed-test pairs, averaging `-0.1134` BPC on Art of War and `-0.1543` on Lincoln. |
| [029 Final Capacity Panel Preregistration](../experiments/029-final-capacity-panel-preregistration.md) | Frozen final protocol | Commits the three-corpus × three-arm × three-seed capacity-control design before source access. |
| [030 Final Capacity Panel and Completion](../experiments/030-final-capacity-panel-and-project-completion.md) | Final evidence and closure | Reports the final near-parameter-matched sealed-test panel and permanently completes the project scope. |

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
- Tokenization: [016](../experiments/016-tokenization-study.md),
  [020](../experiments/020-bpe-context-and-learning-rate.md),
  [021](../experiments/021-boundary-aware-byte-bpe.md),
  [024](../experiments/024-cross-corpus-robustness.md),
  [025](../experiments/025-corpus-by-seed-matrix.md),
  [026](../experiments/026-sealed-test-evaluation.md),
  [027](../experiments/027-hamlet-external-distribution.md),
  [028](../experiments/028-preregistered-external-corpus-panel.md),
  [029](../experiments/029-final-capacity-panel-preregistration.md), and
  [030](../experiments/030-final-capacity-panel-and-project-completion.md).

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

Milestone 021 adds a lossless UTF-8 byte fallback and whitespace-boundary-aware merges. ByteBPE320
reached best BPC `2.0286`; ByteBPE512 reached `2.0083`, the strongest corrected result so far.
The 512-token run's final BPC rose to `2.2450`, making early best-checkpoint selection essential.

Milestone 022 operationalizes that result. Patience-3 early stopping terminates at step 2,500 and
retains best BPC `2.0083`, while stopped-final BPC improves from `2.2450` to `2.0554`. Weight decay
`0.01` reaches best BPC `2.0080`; the `0.00025` difference is too small to interpret as a real gain.

Milestone 023 measures seed sensitivity directly. Across seeds 1337, 2027, and 4242, best BPC is
`2.0225 ± 0.0124` with range `2.0083–2.0384`; best step ranges 1,750–2,250 and stop step
2,500–3,000. The tokenizer result survives all tested seeds, while the observed seed spread confirms
that milestone 022's tiny weight-decay delta was not decision-grade evidence.

Milestone 024 changes the data distribution to a near-size-matched Peter Pan corpus. ByteBPE512
reaches best BPC `2.1539` versus `2.1721` for character, reproducing the direction with a much
smaller 0.83% margin. This supports limited cross-corpus robustness, not a universal tokenizer
advantage; a corpus-by-seed matrix is the next stronger test.

Milestone 025 completes that balanced matrix. ByteBPE512 beats character for seeds 1337, 2027, and
4242 on both Alice and Peter Pan. Its paired mean advantage is `0.0619` BPC on Alice and `0.0252`
on Peter Pan; the `+0.0367` BPC interaction shows that effect magnitude remains corpus-dependent.

Milestone 026 freezes that decision and evaluates new 80/10/10 runs once on terminal test segments.
ByteBPE512 reaches test BPC `2.1178` versus `2.1792` on Alice and `2.2484` versus `2.2742` on Peter
Pan. The direction and approximate margins survive, while every model's test BPC is worse than its
validation BPC.

Milestone 027 preregisters Hamlet before corpus access and transfers the same frozen protocol to a
dramatic play. ByteBPE512 reaches sealed-test BPC `2.2546` versus character's `2.3219`, an advantage
of `0.0673`. Both terminal results are better than validation, reversing milestone 026's gap
direction and reinforcing that chronological difficulty is corpus-dependent.

Milestone 028 expands that confirmatory design to two new corpora and three seeds. ByteBPE512 wins
all six same-seed test comparisons, with mean paired effects `-0.1134 ± 0.0041` BPC on Art of War
and `-0.1543 ± 0.0130` on Lincoln. Every terminal segment is harder than validation, and the
`-0.0409` mean corpus interaction again rejects a universal effect-size interpretation.

Milestone 029 freezes the final capacity-control protocol before source access. Milestone 030
records its complete evidence and closes the project at version 1.0.0; no additional modeling
roadmap remains. The durable lifecycle statement is [Project Completion](project-completion.md).
