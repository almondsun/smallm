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
Experiment 025 completes the 2-tokenizer × 2-corpus × 3-seed matrix. ByteBPE512 wins every paired
comparison. Its mean advantage is `0.0619` BPC on Alice and `0.0252` on Peter Pan, so the direction
is robust within the matrix while the effect magnitude remains corpus-dependent.
Experiment 026 adds a three-way chronological contract and evaluates the frozen decision once.
ByteBPE512 beats character on sealed test BPC by `0.0614` on Alice and `0.0258` on Peter Pan. All
four test results are worse than validation, and these terminal segments are now consumed evidence.
Experiment 027 preregisters a structurally different Hamlet distribution before corpus access.
ByteBPE512 beats character by `0.0673` sealed-test BPC. Both terminal results are better than
validation, demonstrating that chronological gap direction is not stable across corpora.

Experiment 028 preregisters two additional external corpora and three seeds. ByteBPE512 beats the
character control in all six paired sealed-test comparisons, averaging `0.1134` BPC better on Art
of War and `0.1543` better on Lincoln. Use `scripts/summarize_test_matrix.py` for complete balanced
panels; it refuses missing cells, identity mismatches, non-best checkpoints, and partial coverage.

Experiment 030 completes the final capacity-controlled panel. ByteBPE512 beats near-matched
char136 in eight of nine sealed pairs and has lower mean BPC on Frankenstein (`-0.1447`), Douglass
(`-0.0187`), and Origin (`-0.1717`). Douglass seed 4242 reverses by `+0.0018`; the directional
hypothesis succeeds but the all-nine outcome does not. These terminal regions are consumed, and
version 1.0.0 has no further experiment roadmap.

## Artifact Policy

`data/raw/`, `data/processed/`, `checkpoints/`, and `runs/` are local runtime
artifact directories and are ignored by git.
