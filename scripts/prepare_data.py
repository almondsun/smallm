from __future__ import annotations

import argparse

from smallm.config import load_config
from smallm.data import CharTokenizer, load_prepared_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    text = load_prepared_corpus(config.data.prepared_path)
    tokenizer = CharTokenizer.train(text)
    tokenizer.save(config.data.tokenizer_path)
    print(f"saved tokenizer with {tokenizer.vocab_size} tokens to {config.data.tokenizer_path}")


if __name__ == "__main__":
    main()
