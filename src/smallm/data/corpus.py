from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def clean_corpus_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in normalized.split("\n"):
        stripped = line.rstrip()
        is_blank = stripped == ""
        if is_blank and previous_blank:
            continue
        cleaned_lines.append(stripped)
        previous_blank = is_blank
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines) + "\n"


def corpus_stats(
    text: str,
    *,
    train_split: float,
    source_name: str | None,
    source_note: str | None,
    output_path: str | Path,
    top_n: int = 20,
) -> dict[str, Any]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")
    counter = Counter(text)
    total_characters = len(text)
    split_index = int(total_characters * train_split)
    lines = text.splitlines()
    return {
        "source_name": source_name,
        "source_note": source_note,
        "output_path": str(output_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_characters": total_characters,
        "total_lines": len(lines),
        "non_empty_lines": sum(1 for line in lines if line.strip()),
        "unique_characters": len(counter),
        "top_character_frequencies": [
            {"character": char, "count": count}
            for char, count in counter.most_common(top_n)
        ],
        "train_split": train_split,
        "train_characters": split_index,
        "validation_characters": total_characters - split_index,
    }


def load_prepared_corpus(path: str | Path) -> str:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"prepared corpus not found at {corpus_path}. "
            "Run scripts/prepare_corpus.py before preparing data, evaluating baselines, or training."
        )
    return corpus_path.read_text(encoding="utf-8")
