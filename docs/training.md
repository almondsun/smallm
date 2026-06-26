# Training

The training workflow is organized around reproducible local runs. A run starts
from a prepared corpus and ends with a preserved directory containing config,
metrics, checkpoint, summary, generated sample, and dataset provenance.

## Workflow Map

| Step | Command |
| --- | --- |
| Prepare corpus | `python scripts/prepare_corpus.py ...` |
| Prepare tokenizer | `python scripts/prepare_data.py --config configs/<name>.yaml` |
| Evaluate baselines | `python scripts/evaluate_baselines.py --config configs/<name>.yaml` |
| Train | `python scripts/train.py --config configs/<name>.yaml` |
| Inspect run | `python scripts/show_run.py --run latest --run-name <name>` |
| Generate | `python scripts/generate.py --run latest --run-name <name> --prompt "Once"` |

## Configs

| Config | Use |
| --- | --- |
| `configs/smoke.yaml` | Fast full-pipeline check. |
| `configs/gptiny.yaml` | Lightweight GPTiny experiment config used by current milestone reports. |

## Corpus Preparation

Put a plain text corpus at `data/raw/input.txt`.

```bash
python scripts/prepare_corpus.py \
  --input data/raw/input.txt \
  --output data/processed/corpus.txt \
  --stats data/processed/corpus_stats.json \
  --manifest data/processed/corpus_manifest.json \
  --source-name "local text corpus"
```

Use `--source-note` for fetch URLs, extraction notes, or manual curation notes.
The manifest is copied into each training run.

## Smoke Run

```bash
python scripts/prepare_data.py --config configs/smoke.yaml
python scripts/evaluate_baselines.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml
python scripts/show_run.py --run latest --run-name smoke
python scripts/generate.py --run latest --run-name smoke --prompt "Once" --greedy --max-new-tokens 100
```

Use this path to check that the pipeline works after code or environment
changes.

## GPTiny Run

```bash
python scripts/prepare_data.py --config configs/gptiny.yaml
python scripts/evaluate_baselines.py --config configs/gptiny.yaml
python scripts/train.py --config configs/gptiny.yaml
python scripts/show_run.py --run latest --run-name gptiny
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 100
```

This is the main lightweight experiment path.

`model.vocab_size` in config files is a placeholder default for incomplete
configs. Training always uses the tokenizer-derived vocabulary size from the
prepared corpus and stores that actual value in checkpoints and summaries.

## Baseline Evaluation

```bash
python scripts/evaluate_baselines.py --config configs/gptiny.yaml
```

The evaluator prints uniform, unigram, and add-one smoothed bigram validation
loss and perplexity. Use the bigram row as the main simple reference for
GPTiny.

## Run Inspection

```bash
python scripts/list_runs.py
python scripts/list_runs.py --run-name gptiny
python scripts/show_run.py --run runs/gptiny/<run-id>
python scripts/show_run.py --run latest --run-name gptiny
```

`show_run.py` prints run paths, dataset summary fields, the latest metric, and
the saved sample.

## Generation Controls

```bash
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --greedy --max-new-tokens 100
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --temperature 0.8 --seed 1337 --max-new-tokens 100
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --temperature 0.8 --top-k 10 --seed 1337 --max-new-tokens 100
```

Training-time samples use these config fields:

- `sample_max_new_tokens`
- `sample_temperature`
- `sample_top_k`
- `sample_seed`
- `sample_greedy`

The settings are stored in `summary.json` under `generation`.

## Run Directory Contents

| File | Notes |
| --- | --- |
| `config.yaml` | Exact config snapshot. |
| `metrics.jsonl` | Training and validation metrics. |
| `summary.json` | Final losses, paths, dataset fields, and generation settings. |
| `checkpoint.pt` | Model and tokenizer state. |
| `sample.txt` | End-of-training generated sample. |
| `dataset_manifest.json` | Copied corpus manifest. |

## Current Reading Of Results

The current GPTiny model trains, and longer budgets, higher learning rate, and
larger capacity all improve validation loss on the larger public-domain corpus.
Experiment 015 found that wider/deeper variants also improved simple generation
diversity diagnostics, but generated prose still showed phrase-level repetition
and incoherence.

## Artifact Policy

`data/raw/`, `data/processed/`, `checkpoints/`, and `runs/` are local runtime
artifact directories and are ignored by git.
