from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from smallm.config import load_config
from smallm.data import load_prepared_corpus, split_corpus_text, tokenizer_from_state
from smallm.data.corpus import file_sha256
from smallm.evaluation.run_observation import load_run_summary
from smallm.model import GPT, GPTConfig
from smallm.training.artifacts import load_dataset_manifest, verify_dataset_manifest, write_json
from smallm.training.checkpoints import load_checkpoint
from smallm.training.runs import resolve_run_checkpoint
from smallm.training.trainer import evaluate_tokens
from smallm.utils.device import default_device


def evaluate_sealed_test(
    run_dir: Path, *, checkpoint_kind: str = "best"
) -> tuple[Path, dict[str, Any]]:
    config = load_config(run_dir / "config.yaml")
    if config.data.validation_split is None:
        raise ValueError("run does not configure a sealed test split")
    output_path = run_dir / f"test_evaluation_{checkpoint_kind}.json"
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"sealed test evaluation already exists: {output_path}")

    text = load_prepared_corpus(config.data.prepared_path)
    manifest_path = run_dir / "dataset_manifest.json"
    manifest = load_dataset_manifest(manifest_path)
    verify_dataset_manifest(
        manifest,
        prepared_path=config.data.prepared_path,
        prepared_text=text,
        train_split=config.data.train_split,
        validation_split=config.data.validation_split,
    )
    _, _, test_text = split_corpus_text(
        text,
        train_split=config.data.train_split,
        validation_split=config.data.validation_split,
    )
    if len(test_text) <= config.data.block_size:
        raise ValueError("sealed test split is too short for full evaluation")

    checkpoint_path = resolve_run_checkpoint(run_dir, checkpoint_kind)
    checkpoint = load_checkpoint(checkpoint_path)
    summary = load_run_summary(run_dir)
    if summary.get("test_status") != "sealed_unread" or summary.get("test_characters") != len(
        test_text
    ):
        raise ValueError("run summary does not match the sealed test split")
    expected_step = (
        summary.get("best_val_step") if checkpoint_kind == "best" else summary.get("actual_steps")
    )
    if checkpoint.get("step") != expected_step:
        raise ValueError("checkpoint step does not match the completed run summary")
    tokenizer = tokenizer_from_state(checkpoint["tokenizer"])
    token_ids, character_counts = tokenizer.encode_with_character_counts(test_text)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    counts = torch.tensor(character_counts, dtype=torch.long)
    device = default_device()
    model = GPT(GPTConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    result = evaluate_tokens(
        model,
        tokens,
        tokenizer=tokenizer,
        device=device,
        block_size=config.data.block_size,
        max_batches=None,
        character_counts=counts,
    )
    if result is None:
        raise ValueError("sealed test split produced no evaluation targets")
    if result.mode != "full" or result.coverage != 1.0:
        raise ValueError("sealed test evaluation did not cover every target")

    payload = {
        "schema_version": 1,
        "status": "complete",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_step": checkpoint["step"],
        "prepared_sha256": manifest["prepared_sha256"],
        "train_split": config.data.train_split,
        "validation_split": config.data.validation_split,
        "test_characters": len(test_text),
        "test_tokens": len(token_ids),
        "test_loss": result.loss,
        "test_bits_per_character": result.bits_per_character,
        "test_target_tokens": result.target_tokens,
        "test_total_target_tokens": result.total_target_tokens,
        "test_target_characters": result.target_characters,
        "test_coverage": result.coverage,
        "evaluation_mode": result.mode,
    }
    write_json(output_path, payload)
    return output_path, payload
