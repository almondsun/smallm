from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smallm.config import DataConfig


class CharTokenizer:
    """A tiny deterministic character tokenizer for early experiments."""

    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = dict(stoi)
        self.itos = {idx: token for token, idx in self.stoi.items()}

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        return cls({char: idx for idx, char in enumerate(chars)})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def tokenizer_type(self) -> str:
        return "char"

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_state(), indent=2), encoding="utf-8")

    def to_state(self) -> dict[str, Any]:
        return {"type": "char", "stoi": self.stoi, "vocab_size": self.vocab_size}

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state(payload)

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> "CharTokenizer":
        return cls({str(token): int(index) for token, index in payload["stoi"].items()})


def train_tokenizer(config: DataConfig, text: str) -> Any:
    if config.tokenizer_type == "char":
        return CharTokenizer.train(text)
    if config.tokenizer_type == "bpe":
        from smallm.data.bpe_tokenizer import SimpleBPETokenizer

        assert config.bpe_vocab_size is not None
        return SimpleBPETokenizer.train(
            text,
            vocab_size=config.bpe_vocab_size,
            min_frequency=config.bpe_min_frequency,
        )
    raise ValueError(f"unsupported tokenizer type: {config.tokenizer_type}")


def tokenizer_from_state(payload: dict[str, Any]) -> Any:
    tokenizer_type = payload.get("type", "char")
    if tokenizer_type == "char":
        return CharTokenizer.from_state(payload)
    if tokenizer_type == "bpe":
        from smallm.data.bpe_tokenizer import SimpleBPETokenizer

        return SimpleBPETokenizer.from_state(payload)
    raise ValueError(f"unsupported tokenizer artifact type: {tokenizer_type}")


def load_tokenizer(path: str | Path) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tokenizer_from_state(payload)
