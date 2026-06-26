# Training

The initial training path is intentionally small:

1. Put a plain text corpus at `data/raw/input.txt`.
2. Run `python scripts/prepare_data.py --config configs/tiny_gpt.yaml`.
3. Run `python scripts/train.py --config configs/tiny_gpt.yaml`.
4. Generate text with `python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "Once"`.

Runtime artifacts under `data/` and `checkpoints/` are local by default.
