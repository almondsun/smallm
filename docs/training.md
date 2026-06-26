# Training

The training path has two small configs:

- `configs/smoke.yaml` verifies the full pipeline quickly.
- `configs/tiny_gpt.yaml` runs a lightweight tiny GPT training job with visible progress and validation loss.

Smoke test:

1. Put a plain text corpus at `data/raw/input.txt`.
2. Run `python scripts/prepare_data.py --config configs/smoke.yaml`.
3. Run `python scripts/train.py --config configs/smoke.yaml`.
4. Generate text with `python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Once"`.

Tiny training:

```bash
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
```

The training progress logger prints run settings, aligned progress rows, validation loss at evaluation intervals, throughput, ETA, and a final checkpoint summary.

Runtime artifacts under `data/` and `checkpoints/` are local by default.
