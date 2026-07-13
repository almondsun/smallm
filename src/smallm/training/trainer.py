from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass
from math import log
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

import torch
from torch.utils.data import DataLoader

from smallm.config import ExperimentConfig
from smallm.data import TokenBlockDataset, load_prepared_corpus, train_tokenizer
from smallm.generation import generate
from smallm.model import GPT, GPTConfig
from smallm.training.artifacts import (
    MetricsWriter,
    copy_dataset_manifest,
    create_run_dir,
    dataset_summary_from_manifest,
    load_dataset_manifest,
    verify_dataset_manifest,
    write_config_snapshot,
    write_json,
)
from smallm.training.checkpoints import save_checkpoint
from smallm.training.logging import TrainingProgressLogger, TrainingRunInfo
from smallm.utils.device import default_device
from smallm.utils.io import atomic_write_text
from smallm.utils.seed import set_seed


class TokenizerLike(Protocol):
    def source_character_count(self, token_ids: list[int]) -> int: ...

    def encode_with_character_counts(self, text: str) -> tuple[list[int], list[int]]: ...


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int | None,
) -> float | None:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, y)
        if loss is not None:
            losses.extend([float(loss.item())] * y.numel())
    if was_training:
        model.train()
    if not losses:
        return None
    return sum(losses) / len(losses)


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    total_nll: float
    target_tokens: int
    total_target_tokens: int
    target_characters: int
    mode: str

    @property
    def coverage(self) -> float:
        return self.target_tokens / self.total_target_tokens

    @property
    def bits_per_character(self) -> float:
        return self.total_nll / (self.target_characters * log(2))


def _validation_starts(token_count: int, block_size: int, max_batches: int | None) -> list[int]:
    starts = list(range(0, token_count - 1, block_size))
    if max_batches is None or max_batches >= len(starts):
        return starts
    if max_batches == 1:
        return [starts[0]]
    return [
        starts[round(index * (len(starts) - 1) / (max_batches - 1))] for index in range(max_batches)
    ]


@torch.no_grad()
def evaluate_tokens(
    model: GPT,
    tokens: torch.Tensor,
    *,
    tokenizer: TokenizerLike,
    device: torch.device,
    block_size: int,
    max_batches: int | None,
    character_counts: torch.Tensor | None = None,
) -> EvaluationResult | None:
    if tokens.numel() < 2:
        return None
    was_training = model.training
    model.eval()
    starts = _validation_starts(tokens.numel(), block_size, max_batches)
    total_nll = 0.0
    target_tokens = 0
    target_characters = 0
    for start in starts:
        end = min(start + block_size, tokens.numel() - 1)
        x = tokens[start:end].unsqueeze(0).to(device)
        y = tokens[start + 1 : end + 1].unsqueeze(0).to(device)
        _, loss = model(x, y)
        assert loss is not None
        count = y.numel()
        total_nll += float(loss.item()) * count
        target_tokens += count
        if character_counts is None:
            target_characters += tokenizer.source_character_count(y[0].tolist())
        else:
            target_characters += int(character_counts[start + 1 : end + 1].sum().item())
    if was_training:
        model.train()
    return EvaluationResult(
        loss=total_nll / target_tokens,
        total_nll=total_nll,
        target_tokens=target_tokens,
        total_target_tokens=tokens.numel() - 1,
        target_characters=target_characters,
        mode=(
            "full"
            if max_batches is None
            or len(starts) == len(_validation_starts(tokens.numel(), block_size, None))
            else "sampled"
        ),
    )


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
    tokens_seen: int,
    evaluation: EvaluationResult | None = None,
) -> dict[str, float | int | None]:
    elapsed = perf_counter() - start_time
    tokens_per_second = tokens_seen / elapsed if elapsed > 0 else 0.0
    return {
        "schema_version": 2,
        "step": step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_per_second,
        "tokens_seen": tokens_seen,
        "validation_mode": evaluation.mode if evaluation else None,
        "validation_target_tokens": evaluation.target_tokens if evaluation else None,
        "validation_total_target_tokens": evaluation.total_target_tokens if evaluation else None,
        "validation_coverage": evaluation.coverage if evaluation else None,
    }


def train(config: ExperimentConfig) -> Path:
    set_seed(config.train.seed)
    device = default_device()
    text = load_prepared_corpus(config.data.prepared_path)
    dataset_manifest = load_dataset_manifest(config.data.manifest_path)
    verify_dataset_manifest(
        dataset_manifest,
        prepared_path=config.data.prepared_path,
        prepared_text=text,
        train_split=config.data.train_split,
    )
    character_split_index = int(len(text) * config.data.train_split)
    train_text = text[:character_split_index]
    val_text = text[character_split_index:]
    tokenizer = train_tokenizer(config.data, train_text)
    tokenizer.save(config.data.tokenizer_path)
    train_token_ids, _ = tokenizer.encode_with_character_counts(train_text)
    val_token_ids, val_character_counts = tokenizer.encode_with_character_counts(val_text)
    train_tokens = torch.tensor(train_token_ids, dtype=torch.long)
    val_tokens = torch.tensor(val_token_ids, dtype=torch.long)
    val_character_counts_tensor = torch.tensor(val_character_counts, dtype=torch.long)
    train_characters = len(train_text)
    val_characters = len(val_text)
    train_dataset = TokenBlockDataset(train_tokens, config.data.block_size)
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
    run_dir = create_run_dir(config.train.runs_dir, config.train.run_name)
    run_manifest_path, dataset_manifest = copy_dataset_manifest(config.data.manifest_path, run_dir)
    write_config_snapshot(run_dir / "config.yaml", config)

    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=config.model.block_size,
        n_layer=config.model.n_layer,
        n_head=config.model.n_head,
        n_embd=config.model.n_embd,
        dropout=config.model.dropout,
    )
    model = GPT(model_config).to(device)
    optimizer = _build_optimizer(model, config)
    checkpoint_path = run_dir / "checkpoint.pt"
    best_checkpoint_path = run_dir / "best_checkpoint.pt"

    def checkpoint_payload(checkpoint_step: int) -> dict[str, object]:
        return {
            "model_state": model.state_dict(),
            "schema_version": 2,
            "model_config": asdict(model_config),
            "tokenizer": tokenizer.to_state(),
            "tokenizer_path": config.data.tokenizer_path,
            "run_dir": str(run_dir),
            "step": checkpoint_step,
        }

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
    best_evaluation: EvaluationResult | None = None
    start_time = perf_counter()
    tokens_seen = 0
    final_evaluation: EvaluationResult | None = None
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
                tokens_seen += x.numel()
                final_loss = float(loss.item())
                should_log = step % config.train.log_interval == 0 or step == config.train.max_steps
                should_eval = val_tokens.numel() > 1 and step % config.train.eval_interval == 0
                if should_log or should_eval:
                    val_loss = None
                    evaluation = None
                    if should_eval:
                        evaluation = evaluate_tokens(
                            model,
                            val_tokens,
                            tokenizer=tokenizer,
                            device=device,
                            block_size=config.data.block_size,
                            max_batches=config.train.eval_batches,
                            character_counts=val_character_counts_tensor,
                        )
                        val_loss = evaluation.loss if evaluation else None
                        final_evaluation = evaluation
                        final_val_loss = val_loss
                        best_val_loss, best_val_step = _update_best_validation(
                            best_loss=best_val_loss,
                            best_step=best_val_step,
                            val_loss=val_loss,
                            step=step,
                        )
                        if best_val_step == step:
                            best_evaluation = evaluation
                            save_checkpoint(best_checkpoint_path, checkpoint_payload(step))
                    record = _metrics_record(
                        step=step,
                        config=config,
                        start_time=start_time,
                        optimizer=optimizer,
                        train_loss=final_loss,
                        val_loss=val_loss,
                        tokens_seen=tokens_seen,
                        evaluation=evaluation,
                    )
                    logger.progress(
                        step=step,
                        max_steps=config.train.max_steps,
                        train_loss=final_loss,
                        val_loss=val_loss,
                        learning_rate=cast(float, record["learning_rate"]),
                        elapsed_seconds=cast(float, record["elapsed_seconds"]),
                        tokens_per_second=cast(float, record["tokens_per_second"]),
                    )
                    metrics.write(record)
                if step >= config.train.max_steps:
                    break
        needs_final_eval = (
            val_tokens.numel() > 1 and step > 0 and step % config.train.eval_interval != 0
        )
        if needs_final_eval:
            final_evaluation = evaluate_tokens(
                model,
                val_tokens,
                tokenizer=tokenizer,
                device=device,
                block_size=config.data.block_size,
                max_batches=config.train.eval_batches,
                character_counts=val_character_counts_tensor,
            )
            final_val_loss = final_evaluation.loss if final_evaluation else None
            best_val_loss, best_val_step = _update_best_validation(
                best_loss=best_val_loss,
                best_step=best_val_step,
                val_loss=final_val_loss,
                step=step,
            )
            if best_val_step == step:
                best_evaluation = final_evaluation
                save_checkpoint(best_checkpoint_path, checkpoint_payload(step))
            record = _metrics_record(
                step=step,
                config=config,
                start_time=start_time,
                optimizer=optimizer,
                train_loss=final_loss,
                val_loss=final_val_loss,
                tokens_seen=tokens_seen,
                evaluation=final_evaluation,
            )
            logger.progress(
                step=step,
                max_steps=config.train.max_steps,
                train_loss=final_loss,
                val_loss=final_val_loss,
                learning_rate=cast(float, record["learning_rate"]),
                elapsed_seconds=cast(float, record["elapsed_seconds"]),
                tokens_per_second=cast(float, record["tokens_per_second"]),
            )
            metrics.write(record)

    save_checkpoint(checkpoint_path, checkpoint_payload(step))
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
    atomic_write_text(run_dir / "sample.txt", sample_text)
    elapsed_seconds = perf_counter() - start_time
    write_json(
        run_dir / "summary.json",
        {
            "schema_version": 2,
            "status": "complete",
            "corpus_verified": True,
            "run_dir": str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path),
            "best_checkpoint_exists": best_checkpoint_path.exists(),
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
            "tokenizer_type": tokenizer.tokenizer_type,
            "tokenizer_vocab_size": tokenizer.vocab_size,
            "train_tokens": int(train_tokens.numel()),
            "val_tokens": int(val_tokens.numel()),
            "train_characters": train_characters,
            "val_characters": val_characters,
            "final_val_bits_per_char": final_evaluation.bits_per_character
            if final_evaluation
            else None,
            "best_val_bits_per_char": best_evaluation.bits_per_character
            if best_evaluation
            else None,
            "validation_mode": final_evaluation.mode if final_evaluation else None,
            "validation_target_tokens": final_evaluation.target_tokens if final_evaluation else 0,
            "validation_total_target_tokens": final_evaluation.total_target_tokens
            if final_evaluation
            else max(0, val_tokens.numel() - 1),
            "validation_coverage": final_evaluation.coverage if final_evaluation else 0.0,
            "max_steps": config.train.max_steps,
            "device": str(device),
            "environment": {
                "python": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": sys.platform,
                "torch": torch.__version__,
            },
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
