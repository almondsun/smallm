import json
from hashlib import sha256

import pytest

from smallm.data import (
    NORMALIZATION_RULES,
    clean_corpus_text,
    corpus_manifest,
    corpus_stats,
    file_sha256,
    load_prepared_corpus,
)


def test_clean_corpus_text_normalizes_whitespace_conservatively():
    raw = "alpha  \r\n\r\n\r\nbeta\t \n\ngamma"

    cleaned = clean_corpus_text(raw)

    assert cleaned == "alpha\n\nbeta\n\ngamma\n"


def test_corpus_stats_counts_characters_and_split(tmp_path):
    output = tmp_path / "corpus.txt"
    text = "abca\n"

    stats = corpus_stats(
        text,
        train_split=0.6,
        source_name="tiny",
        source_note="test",
        output_path=output,
    )

    assert stats["total_characters"] == 5
    assert stats["total_lines"] == 1
    assert stats["non_empty_lines"] == 1
    assert stats["unique_characters"] == 4
    assert stats["train_characters"] == 3
    assert stats["validation_characters"] == 2
    assert {"character": "a", "count": 2} in stats["top_character_frequencies"]


def test_corpus_stats_json_round_trip(tmp_path):
    stats_path = tmp_path / "stats.json"
    stats = corpus_stats(
        "aa\n",
        train_split=0.5,
        source_name="source",
        source_note=None,
        output_path=tmp_path / "corpus.txt",
    )

    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    loaded = json.loads(stats_path.read_text(encoding="utf-8"))
    assert loaded["source_name"] == "source"
    assert loaded["unique_characters"] == 2


def test_file_sha256_hashes_file_contents(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("abc\n", encoding="utf-8")

    assert file_sha256(path) == sha256(b"abc\n").hexdigest()


def test_corpus_manifest_contains_hashes_and_normalization_rules(tmp_path):
    raw_path = tmp_path / "raw.txt"
    prepared_path = tmp_path / "corpus.txt"
    stats_path = tmp_path / "stats.json"
    raw_path.write_text("alpha\r\n\r\nbeta  ", encoding="utf-8")
    prepared = clean_corpus_text(raw_path.read_text(encoding="utf-8"))
    prepared_path.write_text(prepared, encoding="utf-8")
    stats = corpus_stats(
        prepared,
        train_split=0.5,
        source_name="source",
        source_note="note",
        output_path=prepared_path,
    )
    stats_path.write_text(json.dumps(stats), encoding="utf-8")

    manifest = corpus_manifest(
        raw_path=raw_path,
        prepared_path=prepared_path,
        stats_path=stats_path,
        stats=stats,
        raw_characters=len(raw_path.read_text(encoding="utf-8")),
        source_name="source",
        source_note="note",
    )

    assert manifest["raw_sha256"] == file_sha256(raw_path)
    assert manifest["prepared_sha256"] == file_sha256(prepared_path)
    assert manifest["normalization_rules"] == NORMALIZATION_RULES
    assert manifest["prepared_characters"] == len(prepared)


def test_load_prepared_corpus_missing_file_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run scripts/prepare_corpus.py"):
        load_prepared_corpus(tmp_path / "missing.txt")
