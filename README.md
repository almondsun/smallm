# smaLLM

`smaLLM` is a small GPT-style language model implemented from scratch in PyTorch.

The goal of this project is not to train a frontier-scale model, but to understand the core machinery of modern language models: tokenization, embeddings, causal self-attention, Transformer blocks, next-token prediction, training dynamics, text generation, etc.

## Project layout

- `src/smallm/` contains the reusable Python package.
- `scripts/` contains thin command-line entry points for data preparation, training, and generation.
- `configs/` contains reproducible experiment settings.
- `tests/` contains focused unit tests for tokenizer, dataset, model shape, and generation contracts.
- `docs/` contains project documentation.
- `research/` contains papers, notes, and reference metadata.
- `data/`, `checkpoints/`, and experiment outputs are local runtime artifacts by default.

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/prepare_data.py --config configs/smoke.yaml
python scripts/train.py --config configs/smoke.yaml
python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Once"
```

For a longer lightweight run with visible training progress:

```bash
python scripts/prepare_data.py --config configs/tiny_gpt.yaml
python scripts/train.py --config configs/tiny_gpt.yaml
```

Put a plain text corpus at `data/raw/input.txt` before running the data and training scripts.
