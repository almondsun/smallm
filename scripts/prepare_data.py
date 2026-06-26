from __future__ import annotations

import argparse
from pathlib import Path

from smallm.config import load_config
from smallm.data import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    text = Path(config.data.input_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(text)
    tokenizer.save(config.data.tokenizer_path)
    print(f"saved tokenizer with {tokenizer.vocab_size} tokens to {config.data.tokenizer_path}")


if __name__ == "__main__":
    main()
