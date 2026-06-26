import json

import pytest

from smallm.data import clean_corpus_text, corpus_stats, load_prepared_corpus


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


def test_load_prepared_corpus_missing_file_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run scripts/prepare_corpus.py"):
        load_prepared_corpus(tmp_path / "missing.txt")
