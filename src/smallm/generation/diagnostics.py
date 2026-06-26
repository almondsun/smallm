from __future__ import annotations


def repetition_rate(text: str) -> float:
    """Return the fraction of adjacent character pairs with identical characters."""
    if len(text) < 2:
        return 0.0
    repeated = sum(1 for previous, current in zip(text, text[1:]) if current == previous)
    return repeated / (len(text) - 1)


def distinct_n(text: str, n: int) -> float:
    """Return unique character n-grams divided by total character n-grams."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(text) < n:
        return 0.0
    ngrams = [text[index : index + n] for index in range(len(text) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def longest_repeated_token_run(text: str) -> int:
    """Return the longest run of the same whitespace-delimited token."""
    tokens = text.split()
    if not tokens:
        return 0
    longest = 1
    current = 1
    previous = tokens[0]
    for token in tokens[1:]:
        if token == previous:
            current += 1
        else:
            longest = max(longest, current)
            previous = token
            current = 1
    return max(longest, current)


def generation_diagnostics(text: str) -> dict[str, float | int]:
    return {
        "repetition_rate": repetition_rate(text),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "longest_repeated_token_run": longest_repeated_token_run(text),
    }
