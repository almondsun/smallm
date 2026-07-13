from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smallm.config import DataConfig
from smallm.utils.io import atomic_write_text


class CharTokenizer:
    """A tiny deterministic character tokenizer for early experiments."""

    def __init__(self, stoi: dict[str, int], unk_token: str | None = None) -> None:
        self.stoi = dict(stoi)
        self.itos = {idx: token for token, idx in self.stoi.items()}
        self.unk_token = unk_token
        if unk_token is not None and unk_token not in self.stoi:
            raise ValueError("unk_token must be in stoi")

    @classmethod
    def train(cls, text: str) -> CharTokenizer:
        unk_token = "<unk>"
        chars = sorted(set(text))
        chars.append(unk_token)
        return cls({char: idx for idx, char in enumerate(chars)}, unk_token=unk_token)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @property
    def tokenizer_type(self) -> str:
        return "char"

    def encode(self, text: str) -> list[int]:
        if self.unk_token is None:
            return [self.stoi[char] for char in text]
        return [self.stoi.get(char, self.stoi[self.unk_token]) for char in text]

    def encode_with_character_counts(self, text: str) -> tuple[list[int], list[int]]:
        token_ids = self.encode(text)
        return token_ids, [1] * len(token_ids)

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)

    def source_character_count(self, token_ids: list[int]) -> int:
        return len(token_ids)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(output, json.dumps(self.to_state(), indent=2) + "\n")

    def to_state(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "type": "char",
            "stoi": self.stoi,
            "unk_token": self.unk_token,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_state(payload)

    @classmethod
    def from_state(cls, payload: dict[str, Any]) -> CharTokenizer:
        unk_token = payload.get("unk_token")
        return cls(
            {str(token): int(index) for token, index in payload["stoi"].items()},
            unk_token=str(unk_token) if unk_token is not None else None,
        )


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
    if config.tokenizer_type == "byte_bpe":
        from smallm.data.byte_bpe_tokenizer import ByteBPETokenizer

        assert config.bpe_vocab_size is not None
        return ByteBPETokenizer.train(
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
    if tokenizer_type == "byte_bpe":
        from smallm.data.byte_bpe_tokenizer import ByteBPETokenizer

        return ByteBPETokenizer.from_state(payload)
    raise ValueError(f"unsupported tokenizer artifact type: {tokenizer_type}")


def load_tokenizer(path: str | Path) -> Any:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("tokenizer artifact must contain a mapping")
    from smallm.training.checkpoints import _validate_tokenizer_state

    _validate_tokenizer_state(payload)
    return tokenizer_from_state(payload)
