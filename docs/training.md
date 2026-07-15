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
| Evaluate sealed test | `python scripts/evaluate_test.py --run <run-dir> --checkpoint-kind best` |
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
| `configs/gptiny_char_5k_lr1e-3_earlystop*.yaml` | Three-seed Alice character early-stopping controls. |
| `configs/gptiny_peterpan_*_earlystop_seed*.yaml` | Additional Peter Pan matrix seeds. |
| `configs/gptiny_{alice,peterpan}_{char,bytebpe512}_sealed.yaml` | Frozen 80/10/10 confirmatory runs. |
| `configs/gptiny_hamlet_{char,bytebpe512}_sealed.yaml` | Preregistered 80/10/10 dramatic-play replication. |
| `configs/gptiny_{artofwar,lincoln}_{char,bytebpe512}_sealed_seed{1337,2027,4242}.yaml` | Preregistered 2 × 2 × 3 external-corpus panel. |

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

Add `--validation-split 0.1` with `--train-split 0.8` to reserve a terminal 10% test segment. When
`data.validation_split` is absent, existing configs retain the legacy two-way behavior. With an
explicit validation fraction, tokenizer fitting and training use only the leading train region,
early stopping uses only the middle validation region, and the final region stays unencoded and
unscored.

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

For a run with an explicit sealed split, training records the logical status
`test_status: sealed_unread` and the test character count but no test tokens or metrics. The label
means sealed from modeling and selection; the prepared corpus is still loaded to calculate slice
boundaries. After freezing the decision, evaluate once:

```bash
python scripts/evaluate_test.py --run runs/<name>/<id> --checkpoint-kind best
```

The command verifies the run's copied manifest, corpus checksum, selected checkpoint step, and full
coverage. It writes `test_evaluation_best.json` with checkpoint/corpus hashes, loss, BPC, and exact
target counts, then refuses to overwrite the artifact. Deleting a local artifact can bypass this
guardrail, so the scientific one-shot rule remains procedural.

Aggregate completed runs without selecting a winner:

```bash
python scripts/summarize_runs.py runs/<name-a>/<id> runs/<name-b>/<id> runs/<name-c>/<id>
```

The command reads each run's config and summary, requires distinct training seeds, and prints every
observation plus mean, population standard deviation, minimum, and maximum.

For a balanced two-corpus, two-tokenizer design, label every run explicitly:

```bash
python scripts/summarize_matrix.py \
  --reference-tokenizer char --candidate-tokenizer byte_bpe \
  --cell alice char runs/<name>/<id> \
  --cell alice byte_bpe runs/<name>/<id> \
  --cell peter_pan char runs/<name>/<id> \
  --cell peter_pan byte_bpe runs/<name>/<id>
```

Repeat `--cell` for every seed. The analyzer requires exactly two distinct corpus checksums, both
tokenizers on both corpora, a shared seed set, within-cell experiment fingerprints, and complete
schema-v2 summaries. Reported contrasts are candidate minus reference BPC and are paired by seed.

The final fixed capacity panel uses `scripts/summarize_capacity_panel.py`. Pass every combination
of corpora `frankenstein`, `douglass`, and `origin`; arms `char128`, `char136`, and `bytebpe512`;
and seeds 1337, 2027, and 4242 as `--cell CORPUS ARM RUN_DIR`. The command accepts exactly 27 runs
and validates frozen configurations and the parameter-match tolerance in addition to sealed-test
identity and coverage.

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
| `test_evaluation_best.json` | Optional one-shot sealed-test result; created after training only. |

## Current Reading Of Results

The training and evaluation contracts are complete. Milestone 030 reports the final
capacity-controlled panel: ByteBPE512 beats near-parameter-matched char136 in eight of nine sealed
pairs, with a narrow Douglass reversal and corpus-dependent effect sizes. All terminal regions are
consumed and cannot guide later selection. See the [experiment index](experiments.md) for the full
chronology and [project completion](project-completion.md) for the frozen lifecycle policy.

## Artifact Policy

`data/raw/`, `data/processed/`, `checkpoints/`, and `runs/` are local runtime
artifact directories and are ignored by git.
