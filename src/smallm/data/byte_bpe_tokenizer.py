from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from smallm.utils.io import atomic_write_text

_SEGMENT_PATTERN = re.compile(r"\s+|[^\s]+")
_BASE_VOCAB_SIZE = 256
_MAX_SERIALIZED_TOKEN_LENGTH = 1_024


class ByteBPETokenizer:
    """Deterministic byte-level BPE that never merges across whitespace boundaries."""

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[str, str]],
        min_frequency: int = 2,
    ) -> None:
        self.vocab = dict(vocab)
        self.itos = {index: token for token, index in self.vocab.items()}
        self.merges = list(merges)
        self.min_frequency = min_frequency

    @classmethod
    def train(cls, text: str, vocab_size: int, min_frequency: int = 2) -> ByteBPETokenizer:
        if vocab_size < _BASE_VOCAB_SIZE:
            raise ValueError("byte BPE vocab_size must be at least 256")
        if (
            not isinstance(min_frequency, int)
            or isinstance(min_frequency, bool)
            or not 0 < min_frequency <= 1_000_000_000
        ):
            raise ValueError("min_frequency must be an integer between 1 and 1000000000")

        segments = [_bytes_to_symbols(part.encode("utf-8")) for part in _segments(text)]
        vocab_tokens = [f"{value:02x}" for value in range(_BASE_VOCAB_SIZE)]
        known_tokens = set(vocab_tokens)
        merges: list[tuple[str, str]] = []

        while len(vocab_tokens) < vocab_size:
            pair_counts: Counter[tuple[str, str]] = Counter()
            for symbols in segments:
                pair_counts.update(zip(symbols, symbols[1:]))
            if not pair_counts:
                break
            best_pair, best_count = max(
                pair_counts.items(), key=lambda item: (item[1], item[0][0], item[0][1])
            )
            if best_count < min_frequency:
                break
            merged = "".join(best_pair)
            if merged in known_tokens:
                break
            if len(merged) > _MAX_SERIALIZED_TOKEN_LENGTH:
                break
            segments = [_merge_symbols(symbols, best_pair, merged) for symbols in segments]
            merges.append(best_pair)
            vocab_tokens.append(merged)
            known_tokens.add(merged)

        return cls(
            {token: index for index, token in enumerate(vocab_tokens)},
            merges,
            min_frequency=min_frequency,
        )

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def tokenizer_type(self) -> str:
        return "byte_bpe"

    def encode(self, text: str) -> list[int]:
        return self.encode_with_character_counts(text)[0]

    def encode_with_character_counts(self, text: str) -> tuple[list[int], list[int]]:
        token_ids: list[int] = []
        character_counts: list[int] = []
        for segment in _segments(text):
            is_initial_segment = not token_ids
            symbols = _bytes_to_symbols(segment.encode("utf-8"))
            for left, right in self.merges:
                symbols = _merge_symbols(symbols, (left, right), left + right)
            token_ids.extend(self.vocab[symbol] for symbol in symbols)
            character_counts.extend(
                _character_completion_counts(
                    segment,
                    symbols,
                    exclude_crossing_initial_character=is_initial_segment,
                )
            )
        return token_ids, character_counts

    def decode(self, token_ids: list[int]) -> str:
        raw = b"".join(bytes.fromhex(self.itos[token_id]) for token_id in token_ids)
        return raw.decode("utf-8", errors="replace")

    def source_character_count(self, token_ids: list[int]) -> int:
        return len(self.decode(token_ids))

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "type": "byte_bpe",
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "boundary_policy": "whitespace_segments",
            "encoding": "utf-8",
        }

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> ByteBPETokenizer:
        return cls(
            {str(token): int(index) for token, index in payload["vocab"].items()},
            [(str(left), str(right)) for left, right in payload.get("merges", [])],
            min_frequency=int(payload.get("min_frequency", 2)),
        )

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(output, json.dumps(self.to_state(), indent=2, sort_keys=True) + "\n")


def _segments(text: str) -> list[str]:
    return _SEGMENT_PATTERN.findall(text)


def _bytes_to_symbols(raw: bytes) -> list[str]:
    return [f"{value:02x}" for value in raw]


def _character_completion_counts(
    segment: str,
    symbols: list[str],
    *,
    exclude_crossing_initial_character: bool,
) -> list[int]:
    byte_completions: list[int] = []
    for character in segment:
        encoded = character.encode("utf-8")
        byte_completions.extend([0] * (len(encoded) - 1) + [1])
    counts: list[int] = []
    offset = 0
    for symbol in symbols:
        width = len(symbol) // 2
        counts.append(sum(byte_completions[offset : offset + width]))
        offset += width
    first_token_width = len(symbols[0]) // 2
    character_boundaries: set[int] = set()
    boundary = 0
    for character in segment:
        boundary += len(character.encode("utf-8"))
        character_boundaries.add(boundary)
    if exclude_crossing_initial_character and first_token_width not in character_boundaries:
        for index, count in enumerate(counts[1:], start=1):
            if count:
                counts[index] -= 1
                break
    return counts


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
