# Training

The training path has two small configs:

- `configs/smoke.yaml` verifies the full pipeline quickly.
- `configs/tiny_gpt.yaml` runs a lightweight tiny GPT training job with visible progress and validation loss.

Smoke test:

1. Put a plain text corpus at `data/raw/input.txt`.
2. Run `python scripts/prepare_corpus.py --input data/raw/input.txt --output data/processed/corpus.txt --stats data/processed/corpus_stats.json --manifest data/processed/corpus_manifest.json --source-name "local text corpus"`.
3. Run `python scripts/prepare_data.py --config configs/smoke.yaml`.
4. Run `python scripts/evaluate_baselines.py --config configs/smoke.yaml`.
5. Run `python scripts/train.py --config configs/smoke.yaml`.
6. List runs with `python scripts/list_runs.py`.
7. Inspect the latest smoke run with `python scripts/show_run.py --run latest --run-name smoke`.
8. Generate text with `python scripts/generate.py --run latest --run-name smoke --prompt "Once"`.

Tiny training:

```bash
python scripts/prepare_corpus.py --input data/raw/input.txt --output data/processed/corpus.txt --stats data/processed/corpus_stats.json --manifest data/processed/corpus_manifest.json --source-name "local text corpus"
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/evaluate_baselines.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
python scripts/show_run.py --run latest --run-name tiny_gpt
python scripts/generate.py --run latest --run-name tiny_gpt --prompt "Once"
```

The training progress logger prints run settings, aligned progress rows, validation loss at evaluation intervals, throughput, ETA, and a final checkpoint summary.

Every run writes reproducibility artifacts under `runs/<run-name>/<run-id>/`:

- `config.yaml`: exact config used for the run.
- `metrics.jsonl`: logged training and validation metrics.
- `summary.json`: final losses, duration, parameter count, vocab size, and paths.
- `checkpoint.pt`: model weights and embedded tokenizer state.
- `sample.txt`: generated text from the configured prompt.
- `dataset_manifest.json`: copied corpus manifest with source, checksum, and normalization metadata.

Run discovery:

```bash
python scripts/list_runs.py
python scripts/list_runs.py --run-name smoke
python scripts/show_run.py --run runs/smoke/<run-id>
python scripts/show_run.py --run latest --run-name tiny_gpt
python scripts/generate.py --run runs/smoke/<run-id> --prompt "Once"
python scripts/generate.py --run latest --run-name smoke --prompt "Once"
```

`show_run.py` prints a compact dataset section when the run summary includes manifest metadata.

Runtime artifacts under `data/raw/`, `data/processed/`, `checkpoints/`, and `runs/` are local by default.
