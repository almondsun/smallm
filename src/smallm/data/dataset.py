from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TokenBlockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, tokens: list[int] | torch.Tensor, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be positive")
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.block_size = block_size
        if self.tokens.numel() <= block_size:
            raise ValueError("tokens must contain more items than block_size")

    def __len__(self) -> int:
        return self.tokens.numel() - self.block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[index : index + self.block_size + 1]
        return chunk[:-1], chunk[1:]


def split_tokens(
    tokens: list[int] | torch.Tensor,
    train_split: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0.0 < train_split < 1.0:
        raise ValueError("train_split must be between 0 and 1")
    token_tensor = torch.as_tensor(tokens, dtype=torch.long)
    split_index = int(token_tensor.numel() * train_split)
    return token_tensor[:split_index], token_tensor[split_index:]
