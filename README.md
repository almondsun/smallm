# smaLLM

`smaLLM` is a small, reproducible GPT-style language-model lab built from
scratch in PyTorch. It is meant for inspecting the core language-modeling path:
corpus preparation, character tokenization, causal self-attention, Transformer
blocks, next-token training, baselines, generation, and experiment records.

## Start Here

- [`docs/architecture.md`](docs/architecture.md): package boundaries and model
  data flow.
- [`docs/training.md`](docs/training.md): corpus preparation, baseline
  evaluation, training, run inspection, and generation commands.
- [`docs/experiments.md`](docs/experiments.md): milestone index with results and
  takeaways.
- [`experiments/`](experiments/): chronological experiment reports.

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

The infrastructure is ahead of the model quality. The pipeline is reproducible
and the run records are useful, but the current tiny model remains weak.

Experiment 016 is the clearest status check:

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

The budget and optimizer studies show that the current model was materially
undertrained and benefited from a higher learning rate. The capacity study
shows model size still helps. The first tokenization study shows that simple
BPE is a useful direction, but this BPE128 setup did not beat the character
control; the next technical direction should tune tokenization and training
together before adding unrelated infrastructure.

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
