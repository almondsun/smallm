from __future__ import annotations

import argparse
import json
from pathlib import Path

from smallm.data import clean_corpus_text, corpus_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--source-name")
    parser.add_argument("--source-note")
    parser.add_argument("--train-split", type=float, default=0.9)
    args = parser.parse_args()

    raw_text = Path(args.input).read_text(encoding="utf-8")
    cleaned = clean_corpus_text(raw_text)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    stats = corpus_stats(
        cleaned,
        train_split=args.train_split,
        source_name=args.source_name,
        source_note=args.source_note,
        output_path=output_path,
    )
    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"prepared corpus: {stats['total_characters']} chars, "
        f"{stats['unique_characters']} unique chars -> {output_path}"
    )
    print(f"wrote corpus stats to {stats_path}")


if __name__ == "__main__":
    main()
