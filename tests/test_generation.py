import torch

from smallm.generation import generate
from smallm.model import GPT, GPTConfig


def test_generate_appends_tokens():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=7, block_size=4, n_layer=1, n_head=1, n_embd=8))
    idx = torch.tensor([[1, 2]])

    output = generate(model, idx, max_new_tokens=3)

    assert output.shape == (1, 5)
