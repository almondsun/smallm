from typing import Any

from smallm.data.bpe_tokenizer import SimpleBPETokenizer
from smallm.data.byte_bpe_tokenizer import ByteBPETokenizer
from smallm.data.corpus import (
    NORMALIZATION_RULES,
    clean_corpus_text,
    corpus_manifest,
    corpus_stats,
    extract_gutenberg_body,
    file_sha256,
    load_prepared_corpus,
)
from smallm.data.tokenizer import (
    CharTokenizer,
    load_tokenizer,
    tokenizer_from_state,
    train_tokenizer,
)

__all__ = [
    "NORMALIZATION_RULES",
    "ByteBPETokenizer",
    "CharTokenizer",
    "SimpleBPETokenizer",
    "TokenBlockDataset",
    "clean_corpus_text",
    "corpus_manifest",
    "corpus_stats",
    "extract_gutenberg_body",
    "file_sha256",
    "load_prepared_corpus",
    "load_tokenizer",
    "split_tokens",
    "tokenizer_from_state",
    "train_tokenizer",
]


def __getattr__(name: str) -> Any:
    if name in {"TokenBlockDataset", "split_tokens"}:
        from smallm.data.dataset import TokenBlockDataset, split_tokens

        return {"TokenBlockDataset": TokenBlockDataset, "split_tokens": split_tokens}[name]
    raise AttributeError(f"module 'smallm.data' has no attribute {name!r}")
