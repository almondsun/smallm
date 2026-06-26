from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from smallm.config import ExperimentConfig
from smallm.data import CharTokenizer, TokenBlockDataset, split_tokens
from smallm.generation import generate
from smallm.model import GPT, GPTConfig
from smallm.training.artifacts import (
    MetricsWriter,
    create_run_dir,
    write_config_snapshot,
    write_json,
)
from smallm.training.checkpoints import save_checkpoint
from smallm.training.logging import TrainingProgressLogger, TrainingRunInfo
from smallm.utils.device import default_device
from smallm.utils.seed import set_seed


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int,
) -> float | None:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for batch_index, (x, y) in enumerate(loader):
        if batch_index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            losses.append(float(loss.item()))
    if was_training:
        model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train(config: ExperimentConfig) -> Path:
    set_seed(config.train.seed)
    device = default_device()
    run_dir = create_run_dir(config.train.runs_dir, config.train.run_name)
    write_config_snapshot(run_dir / "config.yaml", config)
    text = Path(config.data.input_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.train(text)
    tokenizer.save(config.data.tokenizer_path)
    train_tokens, val_tokens = split_tokens(tokenizer.encode(text), config.data.train_split)
    train_dataset = TokenBlockDataset(train_tokens, config.data.block_size)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    val_loader = None
    if val_tokens.numel() > config.data.block_size:
        val_dataset = TokenBlockDataset(val_tokens, config.data.block_size)
        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size)

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        dropout=config.model.dropout,
    )
    if config.data.block_size > model_config.block_size:
        raise ValueError("data.block_size cannot exceed model.block_size")
    model = GPT(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    checkpoint_path = run_dir / "checkpoint.pt"
    parameter_count = _parameter_count(model)
    logger = TrainingProgressLogger()
    logger.header(
        TrainingRunInfo(
            project="smaLLM tiny GPT",
            device=str(device),
            vocab_size=tokenizer.vocab_size,
            block_size=model_config.block_size,
            n_layer=model_config.n_layer,
            n_head=model_config.n_head,
            n_embd=model_config.n_embd,
            batch_size=config.train.batch_size,
            max_steps=config.train.max_steps,
            parameter_count=parameter_count,
            checkpoint_path=str(checkpoint_path),
        )
    )

    step = 0
    final_loss = float("nan")
    final_val_loss = None
    start_time = perf_counter()
    with MetricsWriter(run_dir / "metrics.jsonl") as metrics:
        while step < config.train.max_steps:
            for x, y in train_loader:
                model.train()
                x = x.to(device)
                y = y.to(device)
                _, loss = model(x, y)
                assert loss is not None
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                step += 1
                final_loss = float(loss.item())
                should_log = step % config.train.log_interval == 0 or step == config.train.max_steps
                should_eval = (
                    val_loader is not None
                    and config.train.eval_interval > 0
                    and step % config.train.eval_interval == 0
                )
                if should_log or should_eval:
                    val_loss = None
                    if should_eval:
                        val_loss = estimate_loss(
                            model,
                            val_loader,
                            device,
                            config.train.eval_batches,
                        )
                        final_val_loss = val_loss
                    elapsed = perf_counter() - start_time
                    tokens_seen = step * config.train.batch_size * config.data.block_size
                    tokens_per_second = tokens_seen / elapsed if elapsed > 0 else 0.0
                    learning_rate = optimizer.param_groups[0]["lr"]
                    logger.progress(
                        step=step,
                        max_steps=config.train.max_steps,
                        train_loss=final_loss,
                        val_loss=val_loss,
                        learning_rate=learning_rate,
                        elapsed_seconds=elapsed,
                        tokens_per_second=tokens_per_second,
                    )
                    metrics.write(
                        {
                            "step": step,
                            "train_loss": final_loss,
                            "val_loss": val_loss,
                            "learning_rate": learning_rate,
                            "elapsed_seconds": elapsed,
                            "tokens_per_second": tokens_per_second,
                        }
                    )
                if step >= config.train.max_steps:
                    break

    save_checkpoint(
        checkpoint_path,
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model_config),
            "tokenizer": {"stoi": tokenizer.stoi},
            "tokenizer_path": config.data.tokenizer_path,
            "run_dir": str(run_dir),
        },
    )
    prompt = config.train.sample_prompt
    prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    sample_tokens = generate(model, prompt_tokens, config.train.sample_max_new_tokens)
    sample_text = tokenizer.decode(sample_tokens[0].tolist())
    (run_dir / "sample.txt").write_text(sample_text, encoding="utf-8")
    elapsed_seconds = perf_counter() - start_time
    write_json(
        run_dir / "summary.json",
        {
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "sample_path": str(run_dir / "sample.txt"),
            "metrics_path": str(run_dir / "metrics.jsonl"),
            "config_path": str(run_dir / "config.yaml"),
            "final_train_loss": final_loss,
            "final_val_loss": final_val_loss,
            "duration_seconds": elapsed_seconds,
            "parameter_count": parameter_count,
            "vocab_size": tokenizer.vocab_size,
            "max_steps": config.train.max_steps,
            "device": str(device),
        },
    )
    logger.summary(
        checkpoint_path=str(checkpoint_path),
        elapsed_seconds=elapsed_seconds,
        final_loss=final_loss,
    )
    return checkpoint_path
