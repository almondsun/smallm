from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from smallm.utils.io import atomic_write_bytes


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    import io

    buffer = io.BytesIO()
    torch.save(payload, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a mapping")
    schema_version = payload.get("schema_version", 1)
    if schema_version not in {1, 2}:
        raise ValueError(f"unsupported checkpoint schema version: {schema_version}")
    required = {"model_state", "model_config"}
    if schema_version == 2:
        required.update({"tokenizer", "step"})
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {', '.join(missing)}")
    if not isinstance(payload["model_state"], dict) or not isinstance(
        payload["model_config"], dict
    ):
        raise ValueError("checkpoint model_state and model_config must be mappings")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in payload["model_state"].items()
    ):
        raise ValueError("checkpoint model_state must map string names to tensors")
    _validate_model_config(payload["model_config"])
    if schema_version == 2:
        if not isinstance(payload["step"], int) or payload["step"] < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        if not isinstance(payload["tokenizer"], dict):
            raise ValueError("checkpoint tokenizer must be a mapping")
        _validate_tokenizer_state(
            payload["tokenizer"], expected_vocab_size=payload["model_config"]["vocab_size"]
        )
    elif isinstance(payload.get("tokenizer"), dict):
        _validate_tokenizer_state(
            payload["tokenizer"], expected_vocab_size=payload["model_config"]["vocab_size"]
        )
    elif not isinstance(payload.get("tokenizer_path"), str):
        raise ValueError("legacy checkpoint must contain tokenizer state or tokenizer_path")
    return payload


def _validate_model_config(config: dict[str, Any]) -> None:
    required = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"checkpoint model_config is missing keys: {', '.join(missing)}")
    limits = {
        "vocab_size": 1_000_000,
        "block_size": 1_000_000,
        "n_layer": 10_000,
        "n_head": 10_000,
        "n_embd": 1_000_000,
    }
    for field, upper_bound in limits.items():
        value = config[field]
        if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= upper_bound:
            raise ValueError(f"checkpoint model_config.{field} is outside supported bounds")
    dropout = config["dropout"]
    if not isinstance(dropout, int | float) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ValueError("checkpoint model_config.dropout must be in [0, 1)")
    if config["n_embd"] % config["n_head"] != 0:
        raise ValueError("checkpoint model_config.n_embd must be divisible by n_head")
    estimated_parameters = (
        config["vocab_size"] * config["n_embd"] * 2
        + config["block_size"] * config["n_embd"]
        + config["n_layer"] * 12 * config["n_embd"] ** 2
    )
    if estimated_parameters > 250_000_000:
        raise ValueError("checkpoint model_config exceeds the supported parameter budget")


def _validate_tokenizer_state(
    state: dict[str, Any], *, expected_vocab_size: int | None = None
) -> None:
    schema_version = state.get("schema_version", 1)
    if schema_version not in {1, 2}:
        raise ValueError(f"unsupported tokenizer schema version: {schema_version}")
    tokenizer_type = state.get("type", "char")
    vocab = state.get("stoi") if tokenizer_type == "char" else state.get("vocab")
    if tokenizer_type not in {"char", "bpe"} or not isinstance(vocab, dict):
        raise ValueError("checkpoint tokenizer has an unsupported type or vocabulary")
    if not 0 < len(vocab) <= 100_000:
        raise ValueError("checkpoint tokenizer vocabulary is outside supported bounds")
    if not all(isinstance(token, str) and token and len(token) <= 1_024 for token in vocab):
        raise ValueError("checkpoint tokenizer tokens must be bounded non-empty strings")
    ids = list(vocab.values())
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in ids):
        raise ValueError("checkpoint tokenizer IDs must be integers")
    if sorted(ids) != list(range(len(vocab))):
        raise ValueError("checkpoint tokenizer IDs must be unique and contiguous")
    if state.get("vocab_size", len(vocab)) != len(vocab) or (
        expected_vocab_size is not None and len(vocab) != expected_vocab_size
    ):
        raise ValueError("checkpoint tokenizer vocabulary size does not match model config")
    if tokenizer_type == "char":
        unk_token = state.get("unk_token")
        if unk_token is not None and unk_token not in vocab:
            raise ValueError("checkpoint character tokenizer has an invalid unknown token")
        return
    unk_token = state.get("unk_token", "<unk>")
    if unk_token not in vocab:
        raise ValueError("checkpoint BPE tokenizer has an invalid unknown token")
    merges = state.get("merges", [])
    if not isinstance(merges, list) or len(merges) > len(vocab):
        raise ValueError("checkpoint BPE merges are outside supported bounds")
    known: set[str] = {token for token in vocab if len(token) == 1 or token == unk_token}
    for merge in merges:
        if (
            not isinstance(merge, list | tuple)
            or len(merge) != 2
            or not all(isinstance(part, str) and part in known for part in merge)
        ):
            raise ValueError("checkpoint BPE merge references unknown symbols")
        merged = merge[0] + merge[1]
        if merged not in vocab:
            raise ValueError("checkpoint BPE merge result is missing from vocabulary")
        known.add(merged)
