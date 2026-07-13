# smaLLM

[![CI](https://github.com/almondsun/smallm/actions/workflows/ci.yml/badge.svg)](https://github.com/almondsun/smallm/actions/workflows/ci.yml)
[![Python >=3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`smaLLM` is a small, reproducible GPT-style language-model lab built from
scratch in PyTorch. It is meant for inspecting the core language-modeling path:
corpus preparation, character and simple BPE tokenization, causal
self-attention, Transformer blocks, next-token training, baselines, controlled
generation, and experiment records.

## Reviewer Path

For a fast technical review, inspect:

1. [`docs/architecture.md`](docs/architecture.md) for boundaries and the
   end-to-end pipeline.
2. [`docs/experiments.md`](docs/experiments.md) for the milestone index.
3. [`experiments/019-professionalization-and-corrected-evaluation.md`](experiments/019-professionalization-and-corrected-evaluation.md)
   for the latest corrected modeling evidence and its limitations.
4. [`src/smallm/model/`](src/smallm/model/) for the from-scratch GPTiny model.
5. [`src/smallm/training/`](src/smallm/training/) for training and run artifacts.
6. [`src/smallm/data/`](src/smallm/data/) for corpus and tokenizer contracts.
7. [`tests/`](tests/) for focused behavioral checks.
8. [`notes/`](notes/) for the theory, mathematics, and complete system rationale.

This is a reproducible learning and research-engineering artifact, not a
competitive language model.

## Results at a Glance

All neural results below use the 144,530-character Alice corpus and its
chronological 90/10 split unless noted otherwise.

| Milestone | Setup | Main metric | Result | Interpretation |
| --- | --- | --- | --- | --- |
| [011](experiments/011-larger-corpus-tiny-gpt.md) | Add-one bigram baseline | Validation loss | `2.4340` | Strong simple reference on the larger corpus. |
| [011](experiments/011-larger-corpus-tiny-gpt.md) | 500-step GPTiny | Validation loss | `2.5914` | Trailed bigram by `0.1574`; the unchanged budget was too short. |
| [013](experiments/013-gptiny-training-budget-and-optimization.md) | 2k-step GPTiny | Validation loss | `2.2187` | Beat bigram by `0.2153`, showing that training budget mattered. |
| [013](experiments/013-gptiny-training-budget-and-optimization.md) | 5k-step GPTiny control | Validation loss | `1.8601` | Improved local sample texture, but greedy decoding still collapsed. |
| [014](experiments/014-optimizer-and-sampling-diagnostics.md) | 5k-step GPTiny, `lr=0.001` | Best / final validation loss | `1.6501` / `1.6792` | Higher learning rate beat the 5k control; prose remained incoherent. |
| [015](experiments/015-gptiny-capacity-and-generation-diagnostics.md) | Deep character GPTiny | Best validation loss | `1.4950` at step 2500 | Best character validation point; later train/validation separation signaled overfit. |
| [016](experiments/016-tokenization-study.md) | BPE128 vs character control | Estimated best bits/character | `2.4453` vs `2.1569` | BPE shortened validation from 14,453 to 9,522 tokens but underperformed the character control. |
| [017](experiments/017-best-checkpoint-evaluation.md) | Final vs best checkpoint | BPE validation loss and controlled generation | `2.8109` final vs `2.5727` best | Best validation did not improve generation quality under the tested prompt and seed. |
| [019](experiments/019-professionalization-and-corrected-evaluation.md) | Corrected character vs BPE128 | Full-validation best bits/character | `2.0760` char vs `2.0976` BPE128 | BPE128 shortened sequences but remained narrowly worse after removing leakage and coverage bias. |
| [020](experiments/020-bpe-context-and-learning-rate.md) | BPE context/LR controls | Best BPE128 bits/character | `2.0976` remains best | Matched character context and `5e-4` LR changed timing/diversity but did not improve held-out BPC. |

Token-level loss and perplexity are not directly comparable between character
and BPE tokenizers because they predict different units. The tokenizer
comparison therefore uses character-normalized bits per character; milestone 019 computes it from
exact evaluated coverage.

Rows 016–017 use the superseded evaluation contract and are retained as historical evidence; row
019 is the current held-out comparison and computes BPC from exact evaluated coverage.

## Start Here

- [`docs/architecture.md`](docs/architecture.md): package boundaries and model
  data flow.
- [`docs/training.md`](docs/training.md): corpus preparation, baseline
  evaluation, training, run inspection, and generation commands.
- [`docs/experiments.md`](docs/experiments.md): milestone index with results and
  takeaways.
- [`experiments/`](experiments/): chronological experiment reports.
- [`notes/`](notes/): advanced theory and implementation handbook.

## Theory Handbook

The [`notes/`](notes/) handbook derives the autoregressive objective, tokenization, shifted
datasets, causal multi-head attention, Transformer blocks, AdamW, held-out evaluation, BPC,
decoding, diagnostics, provenance, artifact safety, and experiment design. Equations and tensor
shapes link directly to the implementation and its tests.

## Current Capabilities

| Area | What exists |
| --- | --- |
| Corpus preparation | Normalized corpus output, stats, checksums, source metadata, and manifest files. |
| Tokenization | Character-level tokenizer plus a small educational BPE tokenizer for controlled experiments. |
| Model | Decoder-only GPT-style Transformer with causal self-attention. |
| Evaluation | Uniform, unigram, and add-one smoothed bigram baselines. |
| Training | Config-driven training with validation loss, progress logging, checkpoints, metrics, summaries, and samples. |
| Run records | Preserved run directories with copied dataset manifests and selected provenance fields in `summary.json`. |
| Generation | `max_new_tokens`, `temperature`, `top_k`, `seed`, and greedy decoding. |
| Tests | Focused tests for data, baselines, model shape, training artifacts, run utilities, and generation behavior. |

## Quick Start

Install the package:

```bash
python -m pip install -e ".[dev]"
```

Put a plain text corpus at `data/raw/input.txt`, then run the pipeline:

```bash
python scripts/prepare_corpus.py \
  --input data/raw/input.txt \
  --output data/processed/corpus.txt \
  --stats data/processed/corpus_stats.json \
  --manifest data/processed/corpus_manifest.json \
  --source-name "local text corpus"

python scripts/prepare_data.py --config configs/smoke.yaml
python scripts/evaluate_baselines.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml
python scripts/show_run.py --run latest --run-name smoke
python scripts/generate.py --run latest --run-name smoke --prompt "Once" --greedy --max-new-tokens 100
```

For the longer lightweight config:

```bash
python scripts/prepare_data.py --config configs/gptiny.yaml
python scripts/evaluate_baselines.py --config configs/gptiny.yaml
python scripts/train.py --config configs/gptiny.yaml
python scripts/show_run.py --run latest --run-name gptiny
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 100
```

## Repository Map

- [`src/smallm/data/`](src/smallm/data/): corpus preparation, tokenizer, and
  token block dataset.
- [`src/smallm/model/`](src/smallm/model/): GPT config, attention, blocks, and
  language-model head.
- [`src/smallm/evaluation/`](src/smallm/evaluation/): character-level
  baselines.
- [`src/smallm/training/`](src/smallm/training/): trainer, checkpoints,
  artifacts, progress logging, and run discovery.
- [`src/smallm/generation/`](src/smallm/generation/): sampling controls.
- [`scripts/`](scripts/): command-line entry points.
- [`configs/`](configs/): smoke and GPTiny configs.
- [`tests/`](tests/): focused contract tests.

## Current Status

The first public evidence package is complete: the pipeline is reproducible,
the experiment record is inspectable, and automated checks are available
locally and in CI. The current tiny model remains weak, which the reports state
directly.

Experiments 016–017 remain historical evidence, but their tokenizer fitting and sampled-prefix BPC
methodology have been superseded by milestone 019. Their reports carry explicit errata. Corrected
headline results must come from fresh milestone-019 runs rather than mixing metric contracts.

Experiment 017 previously reported:

- Corpus grew from 4,838 to 144,530 prepared characters.
- The larger-corpus bigram baseline reached validation loss `2.4340`.
- The unchanged 500-step GPTiny reached validation loss `2.5914`.
- The 2k-step GPTiny run reached validation loss `2.2187` and beat bigram.
- The 5k-step GPTiny control reached validation loss about `1.860`.
- The 5k-step `lr=0.001` GPTiny run reached final validation loss `1.6792`
  and best validation loss `1.6501`.
- Wider/deeper GPTiny variants improved validation and generation diversity
  diagnostics; the deep variant reached best validation loss `1.4950`.
- Simple BPE128 reduced validation sequence length from 14,453 character tokens
  to 9,522 BPE tokens, but underperformed the character control on estimated
  best bits per character (`2.4453` versus `2.1569`).
- BPE128 produced somewhat more word-like greedy text, but generation still
  showed phrase-level repetition and incoherent prose.
- Training now saves both final and best-validation checkpoints. In the focused
  final-versus-best comparison, the BPE128 best checkpoint improved validation
  loss but produced less diverse text with more phrase reuse under the tested
  greedy and seeded settings. The character best checkpoint also lost diversity
  and did not improve qualitatively.

The budget and optimizer studies show that the model was materially
undertrained and benefited from a higher learning rate. Capacity still helped.
Best-checkpoint evaluation removed an ambiguity from the tokenization study but
did not change its conclusion: this BPE128 setup did not beat the character
control. Any future modeling work should tune tokenization and training
together while evaluating both final and best checkpoints; it is not a
prerequisite for inspecting this release.

## Material Status

Runtime artifacts under `data/raw/`, `data/processed/`, `checkpoints/`, and
`runs/` are local and ignored by default. Experiment reports record selected
results and validation evidence, but generated corpora, checkpoints, and run
outputs are not tracked.

## Not Implemented Yet

- Production-grade tokenizers or external tokenizer libraries.
- Broader tokenization sweeps beyond the first BPE128 comparison.
- Checkpoint resume, mixed precision, distributed training, or dashboards.
- Remote dataset registry or hosted experiment tracking.
