from __future__ import annotations

import argparse

from smallm.config import load_config
from smallm.data import load_prepared_corpus, train_tokenizer
from smallm.evaluation import evaluate_baselines
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
    train_text, val_text = text[:split_index], text[split_index:]
    tokenizer = train_tokenizer(config.data, train_text)
    train_tokens = tokenizer.encode(train_text)
    val_tokens = tokenizer.encode(val_text)
    results = evaluate_baselines(
        train_tokens,
        val_tokens,
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
