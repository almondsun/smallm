import torch
from torch import nn

from smallm.generation import generate
from smallm.model import GPT, GPTConfig


class ScriptedModel(nn.Module):
    def __init__(self, logits: list[list[float]], block_size: int = 8) -> None:
        super().__init__()
        self.config = GPTConfig(vocab_size=len(logits[0]), block_size=block_size)
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.calls = 0

    def forward(self, idx):
        batch_size, sequence_length = idx.shape
        row = self.logits[min(self.calls, self.logits.size(0) - 1)]
        self.calls += 1
        logits = row.view(1, 1, -1).repeat(batch_size, sequence_length, 1)
        return logits.to(idx.device), None


def test_generate_appends_tokens():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=7, block_size=4, n_layer=1, n_head=1, n_embd=8))
    idx = torch.tensor([[1, 2]])

    output = generate(model, idx, max_new_tokens=3)

    assert output.shape == (1, 5)


def test_greedy_generation_is_deterministic():
    idx = torch.tensor([[0]])
    first = generate(ScriptedModel([[0.1, 0.2, 2.0]]), idx, max_new_tokens=4, greedy=True)
    second = generate(ScriptedModel([[0.1, 0.2, 2.0]]), idx, max_new_tokens=4, greedy=True)

    assert first.tolist() == second.tolist()
    assert first.tolist() == [[0, 2, 2, 2, 2]]


def test_seeded_sampling_is_reproducible():
    idx = torch.tensor([[0]])
    logits = [[0.0, 0.2, 0.4, 0.6]]

    first = generate(ScriptedModel(logits), idx, max_new_tokens=8, seed=123)
    second = generate(ScriptedModel(logits), idx, max_new_tokens=8, seed=123)

    assert first.tolist() == second.tolist()


def test_different_seeds_can_produce_different_samples():
    idx = torch.tensor([[0]])
    logits = [[0.0, 0.0, 0.0, 0.0]]

    first = generate(ScriptedModel(logits), idx, max_new_tokens=20, seed=1)
    second = generate(ScriptedModel(logits), idx, max_new_tokens=20, seed=2)

    assert first.tolist() != second.tolist()


def test_top_k_restricts_candidate_tokens():
    idx = torch.tensor([[0]])
    logits = [[0.0, 1.0, 2.0, 3.0]]

    output = generate(ScriptedModel(logits), idx, max_new_tokens=20, top_k=2, seed=123)
    sampled_tokens = output[0, 1:].tolist()

    assert set(sampled_tokens) <= {2, 3}
