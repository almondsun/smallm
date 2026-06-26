from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader

from smallm.config import ExperimentConfig
from smallm.data import CharTokenizer, TokenBlockDataset, load_prepared_corpus, split_tokens
from smallm.generation import generate
from smallm.model import GPT, GPTConfig
from smallm.training.artifacts import (
    MetricsWriter,
    copy_dataset_manifest,
    create_run_dir,
    dataset_summary_from_manifest,
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


def _build_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )


def _update_best_validation(
    *,
    best_loss: float | None,
    best_step: int | None,
    val_loss: float | None,
    step: int,
) -> tuple[float | None, int | None]:
    if val_loss is None:
        return best_loss, best_step
    if best_loss is None or val_loss < best_loss:
        return val_loss, step
    return best_loss, best_step


def _metrics_record(
    *,
    step: int,
    config: ExperimentConfig,
    start_time: float,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float | None,
) -> dict[str, float | int | None]:
    elapsed = perf_counter() - start_time
    tokens_seen = step * config.train.batch_size * config.data.block_size
    tokens_per_second = tokens_seen / elapsed if elapsed > 0 else 0.0
    return {
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
    }


def train(config: ExperimentConfig) -> Path:
    set_seed(config.train.seed)
    device = default_device()
    run_dir = create_run_dir(config.train.runs_dir, config.train.run_name)
    run_manifest_path, dataset_manifest = copy_dataset_manifest(config.data.manifest_path, run_dir)
    write_config_snapshot(run_dir / "config.yaml", config)
    text = load_prepared_corpus(config.data.prepared_path)
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
    optimizer = _build_optimizer(model, config)
    checkpoint_path = run_dir / "checkpoint.pt"
    parameter_count = _parameter_count(model)
    logger = TrainingProgressLogger()
    logger.header(
        TrainingRunInfo(
            project="smaLLM GPTiny",
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
    best_val_loss = None
    best_val_step = None
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
                        best_val_loss, best_val_step = _update_best_validation(
                            best_loss=best_val_loss,
                            best_step=best_val_step,
                            val_loss=val_loss,
                            step=step,
                        )
                    record = _metrics_record(
                        step=step,
                        config=config,
                        start_time=start_time,
                        optimizer=optimizer,
                        train_loss=final_loss,
                        val_loss=val_loss,
                    )
                    logger.progress(
                        step=step,
                        max_steps=config.train.max_steps,
                        train_loss=final_loss,
                        val_loss=val_loss,
                        learning_rate=record["learning_rate"],
                        elapsed_seconds=record["elapsed_seconds"],
                        tokens_per_second=record["tokens_per_second"],
                    )
                    metrics.write(record)
                if step >= config.train.max_steps:
                    break
        needs_final_eval = (
            val_loader is not None
            and config.train.eval_interval > 0
            and step > 0
            and step % config.train.eval_interval != 0
        )
        if needs_final_eval:
            final_val_loss = estimate_loss(
                model,
                val_loader,
                device,
                config.train.eval_batches,
            )
            best_val_loss, best_val_step = _update_best_validation(
                best_loss=best_val_loss,
                best_step=best_val_step,
                val_loss=final_val_loss,
                step=step,
            )
            record = _metrics_record(
                step=step,
                config=config,
                start_time=start_time,
                optimizer=optimizer,
                train_loss=final_loss,
                val_loss=final_val_loss,
            )
            logger.progress(
                step=step,
                max_steps=config.train.max_steps,
                train_loss=final_loss,
                val_loss=final_val_loss,
                learning_rate=record["learning_rate"],
                elapsed_seconds=record["elapsed_seconds"],
                tokens_per_second=record["tokens_per_second"],
            )
            metrics.write(record)

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
    generation_settings = {
        "prompt": prompt,
        "max_new_tokens": config.train.sample_max_new_tokens,
        "temperature": config.train.sample_temperature,
        "top_k": config.train.sample_top_k,
        "seed": config.train.sample_seed,
        "greedy": config.train.sample_greedy,
    }
    sample_tokens = generate(
        model,
        prompt_tokens,
        config.train.sample_max_new_tokens,
        temperature=config.train.sample_temperature,
        top_k=config.train.sample_top_k,
        seed=config.train.sample_seed,
        greedy=config.train.sample_greedy,
    )
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
            "best_val_loss": best_val_loss,
            "best_val_step": best_val_step,
            "duration_seconds": elapsed_seconds,
            "parameter_count": parameter_count,
            "vocab_size": tokenizer.vocab_size,
            "max_steps": config.train.max_steps,
            "device": str(device),
            "generation": generation_settings,
            "dataset": dataset_summary_from_manifest(
                dataset_manifest,
                run_manifest_path=run_manifest_path,
            ),
        },
    )
    logger.summary(
        checkpoint_path=str(checkpoint_path),
        elapsed_seconds=elapsed_seconds,
        final_loss=final_loss,
    )
    return checkpoint_path
