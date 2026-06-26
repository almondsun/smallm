import torch

from smallm.data import TokenBlockDataset, split_tokens


def test_token_block_dataset_returns_shifted_blocks():
    dataset = TokenBlockDataset([1, 2, 3, 4, 5], block_size=3)

    x, y = dataset[0]

    assert torch.equal(x, torch.tensor([1, 2, 3]))
    assert torch.equal(y, torch.tensor([2, 3, 4]))


def test_split_tokens_uses_train_fraction():
    train, val = split_tokens([1, 2, 3, 4, 5], train_split=0.6)

    assert train.tolist() == [1, 2, 3]
    assert val.tolist() == [4, 5]
