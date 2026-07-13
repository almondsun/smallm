import json

import pytest

from smallm.data import CharTokenizer, SimpleBPETokenizer, load_tokenizer


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
