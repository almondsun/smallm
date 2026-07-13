# Training

The training workflow is organized around reproducible local runs. A run starts
from a prepared corpus and ends with a preserved directory containing config,
metrics, final and best-validation checkpoints, summary, generated sample, and
dataset provenance.

## Workflow Map

| Step | Command |
| --- | --- |
| Prepare corpus | `python scripts/prepare_corpus.py ...` |
| Prepare tokenizer | `python scripts/prepare_data.py --config configs/<name>.yaml` |
| Evaluate baselines | `python scripts/evaluate_baselines.py --config configs/<name>.yaml` |
| Train | `python scripts/train.py --config configs/<name>.yaml` |
| Inspect run | `python scripts/show_run.py --run latest --run-name <name>` |
| Generate | `python scripts/generate.py --run latest --run-name <name> --prompt "Once"` |

## Common Make Targets

The top-level `Makefile` keeps routine local commands short:

```bash
make install
make check
make links
make smoke
make generate-smoke
```

`make check` verifies Ruff formatting and linting, strict mypy, pytest with at least 90% coverage,
compileall, and Markdown links. `make smoke` writes an ignored smoke run, and
`make generate-smoke` generates from the latest completed run.

## Configs

| Config | Use |
| --- | --- |
| `configs/smoke.yaml` | Fast full-pipeline check. |
| `configs/gptiny.yaml` | Lightweight GPTiny experiment config used by current milestone reports. |
| `configs/gptiny_5k_lr1e-3_deep.yaml` | Current best character-level capacity control. |
| `configs/gptiny_bpe128_5k_lr1e-3_deep.yaml` | First simple BPE128 tokenization comparison. |
| `configs/gptiny_bpe128_5k_lr1e-3_ctx42.yaml` | Character-context-matched BPE128 control. |
| `configs/gptiny_bpe128_5k_lr5e-4_ctx42.yaml` | Matched-context lower-learning-rate BPE128 run. |
| `configs/gptiny_bpe128_5k_lr5e-4_ctx64.yaml` | Longer-context lower-learning-rate BPE128 run. |
| `configs/gptiny_peterpan_char_5k_lr1e-3_earlystop.yaml` | Peter Pan character control. |
| `configs/gptiny_peterpan_bytebpe512_5k_lr1e-3_ctx37_earlystop.yaml` | Context-matched Peter Pan ByteBPE512 run. |

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

`summary.json` records the configured `max_steps` ceiling and `actual_steps`. Optional
`early_stopping_patience` counts consecutive validation events without an improvement larger than
`early_stopping_min_delta`; `stopped_early`, `stop_reason`, and the terminal state make the decision
inspectable. The final checkpoint is the model at the stop step, while `best_checkpoint.pt` remains
the lowest observed validation-loss model.

Aggregate completed runs without selecting a winner:

```bash
python scripts/summarize_runs.py runs/<name-a>/<id> runs/<name-b>/<id> runs/<name-c>/<id>
```

The command reads each run's config and summary, requires distinct training seeds, and prints every
observation plus mean, population standard deviation, minimum, and maximum.

## Generation Controls

```bash
python scripts/generate.py --run latest --run-name gptiny --prompt "Once" --greedy --max-new-tokens 100
python scripts/generate.py --run latest --run-name gptiny --checkpoint-kind best --prompt "Once" --greedy --max-new-tokens 100
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

Official configs set `eval_batches: null`, which evaluates every non-overlapping validation
target. A positive integer selects evenly distributed deterministic blocks for faster exploratory
runs; summaries record evaluated targets and coverage.

## Run Directory Contents

| File | Notes |
| --- | --- |
| `config.yaml` | Exact config snapshot. |
| `metrics.jsonl` | Training and validation metrics. |
| `summary.json` | Final losses, paths, dataset fields, and generation settings. |
| `checkpoint.pt` | Model and tokenizer state. |
| `best_checkpoint.pt` | Same payload at the best validation step; absent when validation is unavailable. |
| `sample.txt` | End-of-training generated sample. |
| `dataset_manifest.json` | Copied corpus manifest. |

## Current Reading Of Results

The current GPTiny model trains, and longer budgets, higher learning rate, and
larger capacity all improve validation loss on the larger public-domain corpus.
Milestone 019 corrected tokenizer leakage and validation coverage. Full evaluation reached best
BPC `2.0760` for the character control and `2.0976` for BPE128. BPE128 shortened the sequence and
produced competitive seeded diversity, but remained narrowly worse on character-normalized loss.
Experiments 016–017 retain errata and must not be mixed into the corrected metric series.
Experiment 020 found that matching BPE character context and halving its learning rate did not beat
the experiment-019 BPE control. Experiment 021 changed the tokenizer itself: boundary-aware
ByteBPE320 and ByteBPE512 reached best BPC `2.0286` and `2.0083`, both beating the character
control, though ByteBPE512 overfit sharply after step 1,750.
Experiment 022 adds patience-3 early stopping: it ends at step 2,500, halves runtime, and avoids most
final-checkpoint degradation. Weight decay `0.01` changes best BPC by only `0.00025`, which is not
meaningful evidence of improvement.
Experiment 023 repeats the unregularized early-stopping run across seeds 1337, 2027, and 4242.
Best BPC is `2.0225 ± 0.0124` (population SD), and all three runs remain better than the corrected
character control. Stop steps vary from 2,500 to 3,000.
Experiment 024 tests a second book. On the near-size-matched Peter Pan corpus, ByteBPE512 reaches
best BPC `2.1539` versus `2.1721` for character and stops at step 2,750. The 0.83% advantage is a
cross-corpus replication in direction, but too small and sparsely sampled to establish a stable
effect size.

## Artifact Policy

`data/raw/`, `data/processed/`, `checkpoints/`, and `runs/` are local runtime
artifact directories and are ignored by git.
