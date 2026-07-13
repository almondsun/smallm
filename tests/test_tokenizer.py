import json

import pytest

from smallm.config import DataConfig
from smallm.data import (
    ByteBPETokenizer,
    CharTokenizer,
    SimpleBPETokenizer,
    load_tokenizer,
    train_tokenizer,
)


def test_char_tokenizer_round_trip():
    tokenizer = CharTokenizer.train("hello")

    encoded = tokenizer.encode("hello")

    assert tokenizer.decode(encoded) == "hello"
    assert tokenizer.vocab_size == 5
    assert tokenizer.decode(tokenizer.encode("z")) == "<unk>"


def test_bpe_tokenizer_round_trip():
    text = "low lower lowest\n"
    tokenizer = SimpleBPETokenizer.train(text, vocab_size=32)

    encoded = tokenizer.encode(text)

    assert encoded
    assert tokenizer.decode(encoded) == text
    assert all(isinstance(token_id, int) for token_id in encoded)


def test_bpe_tokenizer_save_load_preserves_behavior(tmp_path):
    text = "banana bandana\n"
    path = tmp_path / "tokenizer.json"
    tokenizer = SimpleBPETokenizer.train(text, vocab_size=24)
    tokenizer.save(path)

    loaded = load_tokenizer(path)

    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_bpe_tokenizer_respects_vocab_size_when_possible():
    tokenizer = SimpleBPETokenizer.train("abababab", vocab_size=6, min_frequency=2)

    assert tokenizer.vocab_size <= 6


def test_bpe_tokenizer_rejects_vocab_smaller_than_initial_symbols():
    with pytest.raises(ValueError, match="unique characters"):
        SimpleBPETokenizer.train("abcd", vocab_size=4)


def test_bpe_tokenizer_training_is_deterministic_on_tied_pairs():
    text = "abab cdcd"

    first = SimpleBPETokenizer.train(text, vocab_size=12)
    second = SimpleBPETokenizer.train(text, vocab_size=12)

    assert first.to_state() == second.to_state()


def test_bpe_tokenizer_unknown_characters_use_unk_token():
    tokenizer = SimpleBPETokenizer.train("abc", vocab_size=8)

    encoded = tokenizer.encode("az")

    assert tokenizer.decode(encoded) == "a<unk>"


def test_byte_bpe_round_trips_unicode_and_counts_characters_exactly():
    text = "naïve café 🙂\nnext"
    tokenizer = ByteBPETokenizer.train(text, vocab_size=320)

    token_ids, character_counts = tokenizer.encode_with_character_counts(text)

    assert tokenizer.decode(token_ids) == text
    assert sum(character_counts) == len(text)
    assert len(character_counts) == len(token_ids)


def test_byte_bpe_never_merges_across_whitespace_boundaries():
    tokenizer = ByteBPETokenizer.train("ab ab ab ab", vocab_size=320)

    for left, right in tokenizer.merges:
        raw = bytes.fromhex(left + right)
        assert b" " not in raw or raw.isspace()


def test_byte_bpe_has_total_byte_fallback_and_save_load(tmp_path):
    tokenizer = ByteBPETokenizer.train("ASCII only", vocab_size=320)
    path = tmp_path / "byte_bpe.json"
    tokenizer.save(path)

    loaded = load_tokenizer(path)
    unseen = "漢字 and 🚀"

    assert loaded.decode(loaded.encode(unseen)) == unseen
    assert loaded.to_state() == tokenizer.to_state()


def test_byte_bpe_rejects_vocab_smaller_than_byte_alphabet():
    with pytest.raises(ValueError, match="at least 256"):
        ByteBPETokenizer.train("text", vocab_size=255)


@pytest.mark.parametrize("value", [True, 1.5, 1_000_000_001])
def test_byte_bpe_rejects_invalid_minimum_frequency(value):
    with pytest.raises(ValueError, match="must be an integer"):
        ByteBPETokenizer.train("text", vocab_size=256, min_frequency=value)


def test_tokenizer_factory_builds_configured_byte_bpe():
    tokenizer = train_tokenizer(
        DataConfig(tokenizer_type="byte_bpe", bpe_vocab_size=256),
        "text",
    )

    assert isinstance(tokenizer, ByteBPETokenizer)
    assert tokenizer.vocab_size == 256


def test_byte_bpe_excludes_character_crossing_initial_context_token():
    tokenizer = ByteBPETokenizer.train("é🙂", vocab_size=256)

    _, character_counts = tokenizer.encode_with_character_counts("é🙂")

    assert sum(character_counts) == 1


def test_byte_bpe_excludes_crossing_after_complete_initial_character():
    vocab = {f"{value:02x}": value for value in range(256)}
    vocab["61c3"] = 256
    tokenizer = ByteBPETokenizer(vocab, [("61", "c3")])

    token_ids, character_counts = tokenizer.encode_with_character_counts("aé")

    assert [tokenizer.itos[token_id] for token_id in token_ids] == ["61c3", "a9"]
    assert character_counts == [1, 0]


def test_load_tokenizer_supports_legacy_char_artifact(tmp_path):
    path = tmp_path / "legacy_char.json"
    path.write_text('{"stoi": {"a": 0, "b": 1}}', encoding="utf-8")

    tokenizer = load_tokenizer(path)

    assert tokenizer.decode(tokenizer.encode("ab")) == "ab"


def test_load_tokenizer_rejects_sparse_ids(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"type": "char", "stoi": {"a": 0, "b": 2}}), encoding="utf-8")

    with pytest.raises(ValueError, match="contiguous"):
        load_tokenizer(path)
