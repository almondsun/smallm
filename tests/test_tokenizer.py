from smallm.data import CharTokenizer


def test_char_tokenizer_round_trip():
    tokenizer = CharTokenizer.train("hello")

    encoded = tokenizer.encode("hello")

    assert tokenizer.decode(encoded) == "hello"
    assert tokenizer.vocab_size == 4
