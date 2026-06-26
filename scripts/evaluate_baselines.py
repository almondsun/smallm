from __future__ import annotations

import argparse
from pathlib import Path

from smallm.config import load_config
from smallm.data import CharTokenizer, split_tokens
from smallm.evaluation import evaluate_baselines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    text = Path(config.data.input_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(text)
    train_tokens, val_tokens = split_tokens(tokenizer.encode(text), config.data.train_split)
    results = evaluate_baselines(
        train_tokens.tolist(),
        val_tokens.tolist(),
        tokenizer.vocab_size,
    )

    print(f"{'baseline':<10} {'val_loss':>10} {'perplexity':>12} notes")
    print("-" * 80)
    for result in results:
        print(
            f"{result.name:<10} "
            f"{result.validation_loss:>10.4f} "
            f"{result.perplexity:>12.2f} "
            f"{result.notes}"
        )


if __name__ == "__main__":
    main()
