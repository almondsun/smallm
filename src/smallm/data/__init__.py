from smallm.data.corpus import clean_corpus_text, corpus_stats, load_prepared_corpus
from smallm.data.tokenizer import CharTokenizer

__all__ = [
    "CharTokenizer",
    "TokenBlockDataset",
    "clean_corpus_text",
    "corpus_stats",
    "load_prepared_corpus",
    "split_tokens",
]


def __getattr__(name: str):
    if name in {"TokenBlockDataset", "split_tokens"}:
        from smallm.data.dataset import TokenBlockDataset, split_tokens

        return {"TokenBlockDataset": TokenBlockDataset, "split_tokens": split_tokens}[name]
    raise AttributeError(f"module 'smallm.data' has no attribute {name!r}")
