from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp, isinf, log


@dataclass(frozen=True)
class BaselineResult:
    name: str
    validation_loss: float
    perplexity: float
    notes: str


def perplexity(loss: float) -> float:
    if isinf(loss):
        return float("inf")
    return exp(loss)


def _mean_negative_log_likelihood(probs: list[float]) -> float:
    if not probs:
        raise ValueError("validation sequence must not be empty")
    if any(prob <= 0.0 for prob in probs):
        return float("inf")
    return -sum(log(prob) for prob in probs) / len(probs)


def uniform_loss(vocab_size: int, validation_length: int) -> float:
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if validation_length < 1:
        raise ValueError("validation_length must be positive")
    return log(vocab_size)


def unigram_loss(train_tokens: list[int], val_tokens: list[int]) -> float:
    if not train_tokens:
        raise ValueError("train_tokens must not be empty")
    counts = Counter(train_tokens)
    total = len(train_tokens)
    probs = [counts[token] / total for token in val_tokens]
    return _mean_negative_log_likelihood(probs)


def bigram_loss(
    train_tokens: list[int],
    val_tokens: list[int],
    vocab_size: int,
    smoothing: float = 1.0,
) -> float:
    if len(train_tokens) < 2:
        raise ValueError("train_tokens must contain at least two tokens")
    if not val_tokens:
        raise ValueError("val_tokens must not be empty")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if smoothing <= 0.0:
        raise ValueError("smoothing must be positive")

    context_counts = Counter(train_tokens[:-1])
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for prev_token, next_token in zip(train_tokens[:-1], train_tokens[1:]):
        transition_counts[prev_token][next_token] += 1

    probs: list[float] = []
    context = train_tokens[-1]
    for token in val_tokens:
        numerator = transition_counts[context][token] + smoothing
        denominator = context_counts[context] + smoothing * vocab_size
        probs.append(numerator / denominator)
        context = token
    return _mean_negative_log_likelihood(probs)


def evaluate_baselines(
    train_tokens: list[int],
    val_tokens: list[int],
    vocab_size: int,
    smoothing: float = 1.0,
) -> list[BaselineResult]:
    uniform = uniform_loss(vocab_size, len(val_tokens))
    unigram = unigram_loss(train_tokens, val_tokens)
    bigram = bigram_loss(train_tokens, val_tokens, vocab_size, smoothing)
    return [
        BaselineResult("uniform", uniform, perplexity(uniform), "equal probability for every character"),
        BaselineResult("unigram", unigram, perplexity(unigram), "train-set character frequencies"),
        BaselineResult("bigram", bigram, perplexity(bigram), f"add-{smoothing:g} smoothed character transitions"),
    ]
