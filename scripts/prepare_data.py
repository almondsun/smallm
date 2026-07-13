from __future__ import annotations

import argparse

from smallm.config import load_config
from smallm.data import load_prepared_corpus, train_tokenizer
from smallm.training.artifacts import load_dataset_manifest, verify_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    text = load_prepared_corpus(config.data.prepared_path)
    verify_dataset_manifest(
        load_dataset_manifest(config.data.manifest_path),
        prepared_path=config.data.prepared_path,
        prepared_text=text,
        train_split=config.data.train_split,
    )
    split_index = int(len(text) * config.data.train_split)
    tokenizer = train_tokenizer(config.data, text[:split_index])
    tokenizer.save(config.data.tokenizer_path)
    label = {
        "bpe": "BPE",
        "byte_bpe": "byte BPE",
        "char": "character",
    }[config.data.tokenizer_type]
    print(
        f"saved {label} tokenizer with {tokenizer.vocab_size} tokens to {config.data.tokenizer_path}"
    )


if __name__ == "__main__":
    main()
