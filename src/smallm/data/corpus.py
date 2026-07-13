from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NORMALIZATION_RULES = [
    "normalize CRLF/CR to LF",
    "strip trailing whitespace",
    "collapse repeated blank lines",
    "ensure final newline",
]

_GUTENBERG_START = re.compile(r"^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\*$")
_GUTENBERG_END = re.compile(r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .+ \*\*\*$")


def extract_gutenberg_body(text: str, *, max_characters: int | None = None) -> str:
    if max_characters is not None and max_characters <= 0:
        raise ValueError("max_characters must be positive or null")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if _GUTENBERG_START.fullmatch(line.strip())]
    ends = [index for index, line in enumerate(lines) if _GUTENBERG_END.fullmatch(line.strip())]
    if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
        raise ValueError("expected exactly one ordered Project Gutenberg START/END marker pair")
    body = "".join(lines[starts[0] + 1 : ends[0]]).lstrip("\n")
    if not body.strip():
        raise ValueError("Project Gutenberg body is empty")
    return body if max_characters is None else body[:max_characters]


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            {"character": char, "count": count} for char, count in counter.most_common(top_n)
        ],
        "train_split": train_split,
        "train_characters": split_index,
        "validation_characters": total_characters - split_index,
    }


def corpus_manifest(
    *,
    raw_path: str | Path,
    prepared_path: str | Path,
    stats_path: str | Path,
    stats: dict[str, Any],
    raw_characters: int,
    source_name: str | None,
    source_note: str | None,
) -> dict[str, Any]:
    return {
        "source_name": source_name,
        "source_note": source_note,
        "raw_path": str(raw_path),
        "prepared_path": str(prepared_path),
        "stats_path": str(stats_path),
        "raw_sha256": file_sha256(raw_path),
        "prepared_sha256": file_sha256(prepared_path),
        "raw_characters": raw_characters,
        "prepared_characters": stats["total_characters"],
        "unique_characters": stats["unique_characters"],
        "train_split": stats["train_split"],
        "train_characters": stats["train_characters"],
        "validation_characters": stats["validation_characters"],
        "normalization_rules": NORMALIZATION_RULES,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def load_prepared_corpus(path: str | Path) -> str:
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"prepared corpus not found at {corpus_path}. "
            "Run scripts/prepare_corpus.py before preparing data, evaluating baselines, or training."
        )
    return corpus_path.read_text(encoding="utf-8")
