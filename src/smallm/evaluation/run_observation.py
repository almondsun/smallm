from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any

from smallm.config import load_config
from smallm.evaluation.robustness import RunObservation

_MAX_ARTIFACT_BYTES = 1_000_000


def _read_bounded_text(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            payload = handle.read(_MAX_ARTIFACT_BYTES + 1)
    except OSError as exc:
        raise ValueError(f"required run artifact cannot be read: {path}: {exc}") from exc
    if len(payload) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"run artifact exceeds {_MAX_ARTIFACT_BYTES} bytes: {path}")
    return payload.decode("utf-8")


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    try:
        payload = json.loads(_read_bounded_text(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run summary at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"run summary must be a mapping: {path}")
    if payload.get("schema_version") != 2 or payload.get("status") != "complete":
        raise ValueError(f"run summary is not a complete schema-v2 artifact: {path}")
    return payload


def _required_float(summary: dict[str, Any], field: str, *, positive: bool = False) -> float:
    value = summary.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"run summary field {field!r} must be numeric")
    if isinstance(value, int) and abs(value) > 1_000_000_000:
        raise ValueError(f"run summary field {field!r} is outside supported bounds")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"run summary field {field!r} is outside supported bounds") from exc
    if not isfinite(number) or number < 0 or (positive and number == 0) or number > 1_000_000_000:
        raise ValueError(f"run summary field {field!r} is outside supported bounds")
    return number


def _required_int(summary: dict[str, Any], field: str) -> int:
    value = summary.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 1_000_000_000:
        raise ValueError(f"run summary field {field!r} must be a bounded non-negative integer")
    return value


def _comparison_fingerprint(config: Any, summary: dict[str, Any]) -> str:
    config_state = asdict(config)
    train_state = config_state["train"]
    for field in ("run_name", "runs_dir", "seed"):
        train_state.pop(field)
    dataset = summary.get("dataset")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("prepared_sha256"), str):
        raise ValueError("run summary dataset must contain prepared_sha256")
    identity = {
        "config": config_state,
        "dataset": {
            "prepared_sha256": dataset["prepared_sha256"],
            "train_split": dataset.get("train_split"),
            "train_characters": dataset.get("train_characters"),
            "validation_characters": dataset.get("validation_characters"),
        },
        "summary_schema_version": summary["schema_version"],
        "validation_mode": summary.get("validation_mode"),
        "validation_coverage": summary.get("validation_coverage"),
        "tokenizer_type": summary.get("tokenizer_type"),
        "tokenizer_vocab_size": summary.get("tokenizer_vocab_size"),
    }
    encoded = json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _observation_from_summary(run_dir: Path, summary: dict[str, Any]) -> RunObservation:
    config_path = run_dir / "config.yaml"
    try:
        _read_bounded_text(config_path)
        config = load_config(config_path)
    except (OSError, UnicodeError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(f"invalid run config at {config_path}: {exc}") from exc
    actual_steps = _required_int(summary, "actual_steps")
    best_step = _required_int(summary, "best_val_step")
    if best_step > actual_steps:
        raise ValueError("run summary best_val_step cannot exceed actual_steps")
    return RunObservation(
        seed=config.train.seed,
        actual_steps=actual_steps,
        best_step=best_step,
        best_bpc=_required_float(summary, "best_val_bits_per_char", positive=True),
        final_bpc=_required_float(summary, "final_val_bits_per_char", positive=True),
        duration_seconds=_required_float(summary, "duration_seconds"),
        comparison_fingerprint=_comparison_fingerprint(config, summary),
    )


def load_run_artifact(run_dir: Path) -> tuple[dict[str, Any], RunObservation]:
    summary = load_run_summary(run_dir)
    return summary, _observation_from_summary(run_dir, summary)


def load_run_observation(run_dir: Path) -> RunObservation:
    return load_run_artifact(run_dir)[1]
