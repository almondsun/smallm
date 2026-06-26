import pytest

from smallm.generation import (
    distinct_n,
    generation_diagnostics,
    longest_repeated_token_run,
    repetition_rate,
)


def test_empty_text_has_zero_diagnostics():
    assert repetition_rate("") == 0.0
    assert distinct_n("", 1) == 0.0
    assert longest_repeated_token_run("") == 0
    assert generation_diagnostics("") == {
        "repetition_rate": 0.0,
        "distinct_1": 0.0,
        "distinct_2": 0.0,
        "longest_repeated_token_run": 0,
    }


def test_single_character_text_has_zero_repetition_and_distinct_one():
    assert repetition_rate("a") == 0.0
    assert distinct_n("a", 1) == 1.0
    assert distinct_n("a", 2) == 0.0


def test_repeated_characters_increase_repetition_rate():
    assert repetition_rate("aaaa") == 1.0
    assert distinct_n("aaaa", 1) == 0.25
    assert distinct_n("aaaa", 2) == 1 / 3


def test_longest_repeated_token_run_counts_words():
    assert longest_repeated_token_run("the the the cat") == 3
    assert longest_repeated_token_run("the cat cat sat sat sat") == 3


def test_mixed_text_diagnostics_are_bounded():
    diagnostics = generation_diagnostics("Once upon a time")

    assert 0.0 <= diagnostics["repetition_rate"] <= 1.0
    assert 0.0 <= diagnostics["distinct_1"] <= 1.0
    assert 0.0 <= diagnostics["distinct_2"] <= 1.0
    assert diagnostics["longest_repeated_token_run"] == 1


def test_distinct_n_rejects_non_positive_n():
    with pytest.raises(ValueError, match="n must be positive"):
        distinct_n("abc", 0)
