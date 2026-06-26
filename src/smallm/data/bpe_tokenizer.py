from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


class SimpleBPETokenizer:
    """A small deterministic character-level BPE tokenizer for experiments."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        unk_token: str = "<unk>",
        min_frequency: int = 2,
    ) -> None:
        self.vocab = dict(vocab)
        self.itos = {idx: token for token, idx in self.vocab.items()}
        self.merges = list(merges)
        self.unk_token = unk_token
        self.min_frequency = min_frequency
        if self.unk_token not in self.vocab:
            raise ValueError("unk_token must be in vocab")

    @classmethod
    def train(cls, text: str, vocab_size: int, min_frequency: int = 2) -> "SimpleBPETokenizer":
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")

        unk_token = "<unk>"
        symbols = list(text)
        vocab_tokens = sorted(set(symbols))
        if unk_token not in vocab_tokens:
            vocab_tokens.append(unk_token)
        if vocab_size < len(vocab_tokens):
            raise ValueError(
                "vocab_size must be at least the number of unique characters plus the unk token"
            )
        merges: list[tuple[str, str]] = []
        known_tokens = set(vocab_tokens)

        while len(vocab_tokens) < vocab_size:
            pair_counts = Counter(zip(symbols, symbols[1:]))
            if not pair_counts:
                break
            best_pair, best_count = max(
                pair_counts.items(),
                key=lambda item: (item[1], item[0][0], item[0][1]),
            )
            if best_count < min_frequency:
                break
            merged = "".join(best_pair)
            if merged in known_tokens:
                break

            symbols = _merge_symbols(symbols, best_pair, merged)
            merges.append(best_pair)
            vocab_tokens.append(merged)
            known_tokens.add(merged)

        vocab = {token: index for index, token in enumerate(vocab_tokens)}
        return cls(vocab, merges, unk_token=unk_token, min_frequency=min_frequency)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def tokenizer_type(self) -> str:
        return "bpe"

    def encode(self, text: str) -> list[int]:
        symbols = [char if char in self.vocab else self.unk_token for char in text]
        for left, right in self.merges:
            merged = left + right
            symbols = _merge_symbols(symbols, (left, right), merged)
        return [self.vocab.get(symbol, self.vocab[self.unk_token]) for symbol in symbols]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in ids)

    def to_state(self) -> dict[str, Any]:
        return {
            "type": "bpe",
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> "SimpleBPETokenizer":
        return cls(
            {str(token): int(index) for token, index in payload["vocab"].items()},
            [(str(left), str(right)) for left, right in payload.get("merges", [])],
            unk_token=str(payload.get("unk_token", "<unk>")),
            min_frequency=int(payload.get("min_frequency", 2)),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_state(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SimpleBPETokenizer":
        return cls.from_state(json.loads(Path(path).read_text(encoding="utf-8")))


def _merge_symbols(symbols: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    output: list[str] = []
    index = 0
    while index < len(symbols):
        if index + 1 < len(symbols) and (symbols[index], symbols[index + 1]) == pair:
            output.append(merged)
            index += 2
        else:
            output.append(symbols[index])
            index += 1
    return output
