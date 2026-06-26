import torch

from smallm.model import GPT, GPTConfig


def test_gpt_forward_returns_logits_and_loss():
    model = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8))
    idx = torch.randint(0, 11, (2, 5))

    logits, loss = model(idx, idx)

    assert logits.shape == (2, 5, 11)
    assert loss is not None


def test_attention_mask_is_causal():
    model = GPT(GPTConfig(vocab_size=11, block_size=8, n_layer=1, n_head=1, n_embd=8))
    mask = model.blocks[0].attn.causal_mask[0, 0, :4, :4]

    assert torch.equal(mask, torch.tril(torch.ones(4, 4, dtype=torch.bool)))
