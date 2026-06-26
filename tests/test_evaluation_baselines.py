from math import isclose, log

from smallm.evaluation.baselines import bigram_loss, evaluate_baselines, perplexity, uniform_loss, unigram_loss


def test_uniform_loss_is_log_vocab_size():
    loss = uniform_loss(vocab_size=4, validation_length=3)

    assert isclose(loss, log(4))
    assert isclose(perplexity(loss), 4.0)


def test_unigram_loss_uses_train_token_frequencies():
    loss = unigram_loss(train_tokens=[0, 0, 1, 1], val_tokens=[0, 1])

    assert isclose(loss, -((log(0.5) + log(0.5)) / 2))


def test_bigram_loss_uses_smoothed_transition_probabilities():
    loss = bigram_loss(
        train_tokens=[0, 1, 0, 1],
        val_tokens=[0, 1],
        vocab_size=2,
        smoothing=1.0,
    )

    expected_first = 2 / 3
    expected_second = 3 / 4
    assert isclose(loss, -((log(expected_first) + log(expected_second)) / 2))


def test_evaluate_baselines_returns_expected_rows():
    results = evaluate_baselines(
        train_tokens=[0, 0, 1, 1, 0],
        val_tokens=[1, 0],
        vocab_size=2,
    )

    assert [result.name for result in results] == ["uniform", "unigram", "bigram"]
    assert all(result.validation_loss > 0 for result in results)
