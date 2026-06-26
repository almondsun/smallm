from smallm.data.tokenizer import CharTokenizer

__all__ = ["CharTokenizer", "TokenBlockDataset", "split_tokens"]


def __getattr__(name: str):
    if name in {"TokenBlockDataset", "split_tokens"}:
        from smallm.data.dataset import TokenBlockDataset, split_tokens

        return {"TokenBlockDataset": TokenBlockDataset, "split_tokens": split_tokens}[name]
    raise AttributeError(f"module 'smallm.data' has no attribute {name!r}")
