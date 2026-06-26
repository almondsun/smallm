from __future__ import annotations

import torch
from torch.nn import functional as F

from smallm.model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    model.eval()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=idx.device)
        generator.manual_seed(seed)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if greedy:
            next_idx = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_idx), dim=1)
            continue

        logits = logits / temperature
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1, generator=generator)
        idx = torch.cat((idx, next_idx), dim=1)
    return idx
