from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smallm.config import ExperimentConfig
from smallm.data import CharTokenizer, TokenBlockDataset, split_tokens
from smallm.model import GPT, GPTConfig
from smallm.training.checkpoints import save_checkpoint
from smallm.utils.device import default_device
from smallm.utils.seed import set_seed


def train(config: ExperimentConfig) -> Path:
    set_seed(config.train.seed)
    device = default_device()
    text = Path(config.data.input_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(text)
    tokenizer.save(config.data.tokenizer_path)
    train_tokens, _ = split_tokens(tokenizer.encode(text), config.data.train_split)
    dataset = TokenBlockDataset(train_tokens, config.data.block_size)
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True)

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        dropout=config.model.dropout,
    )
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)

    step = 0
    while step < config.train.max_steps:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            assert loss is not None
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step >= config.train.max_steps:
                break

    checkpoint_path = Path(config.train.checkpoint_dir) / "latest.pt"
    save_checkpoint(
        checkpoint_path,
        {
            "model_state": model.state_dict(),
            "model_config": model_config,
            "tokenizer_path": config.data.tokenizer_path,
        },
    )
    return checkpoint_path
