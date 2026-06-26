# Training

The training path has two small configs:

- `configs/smoke.yaml` verifies the full pipeline quickly.
- `configs/tiny_gpt.yaml` runs a lightweight tiny GPT training job with visible progress and validation loss.

Smoke test:

1. Put a plain text corpus at `data/raw/input.txt`.
2. Run `python scripts/prepare_data.py --config configs/smoke.yaml`.
3. Run `python scripts/train.py --config configs/smoke.yaml`.
4. Generate text with `python scripts/generate.py --checkpoint runs/smoke/<run-id>/checkpoint.pt --prompt "Once"`.

Tiny training:

```bash
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
```

The training progress logger prints run settings, aligned progress rows, validation loss at evaluation intervals, throughput, ETA, and a final checkpoint summary.

Every run writes reproducibility artifacts under `runs/<run-name>/<run-id>/`:

- `config.yaml`: exact config used for the run.
- `metrics.jsonl`: logged training and validation metrics.
- `summary.json`: final losses, duration, parameter count, vocab size, and paths.
- `checkpoint.pt`: model weights and embedded tokenizer state.
- `sample.txt`: generated text from the configured prompt.

Runtime artifacts under `data/`, `checkpoints/`, and `runs/` are local by default.
