from __future__ import annotations

import argparse

from smallm.config import load_config
from smallm.data import load_prepared_corpus, train_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    text = load_prepared_corpus(config.data.prepared_path)
    tokenizer = train_tokenizer(config.data, text)
    tokenizer.save(config.data.tokenizer_path)
    label = "BPE" if config.data.tokenizer_type == "bpe" else "char"
    print(f"saved {label} tokenizer with {tokenizer.vocab_size} tokens to {config.data.tokenizer_path}")


if __name__ == "__main__":
    main()
