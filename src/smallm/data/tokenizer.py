from __future__ import annotations

import json
from pathlib import Path


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

    def encode(self, text: str) -> list[int]:
        return [self.stoi[char] for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[token_id] for token_id in token_ids)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"stoi": self.stoi}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(payload["stoi"])
