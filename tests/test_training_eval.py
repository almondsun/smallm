import torch
from torch.utils.data import DataLoader

from smallm.data import TokenBlockDataset
from smallm.model import GPT, GPTConfig
from smallm.training.trainer import estimate_loss


def test_estimate_loss_returns_loss_and_restores_train_mode():
    model = GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=8))
    model.train()
    dataset = TokenBlockDataset([0, 1, 2, 3, 4, 5, 6, 7], block_size=4)
    loader = DataLoader(dataset, batch_size=2)

    loss = estimate_loss(model, loader, torch.device("cpu"), max_batches=1)

    assert loss is not None
    assert loss > 0
    assert model.training
