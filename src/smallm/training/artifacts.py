from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any, cast

from smallm.config import ExperimentConfig
from smallm.utils.io import atomic_write_text

DATASET_SUMMARY_FIELDS = [
    "source_name",
    "source_note",
    "raw_sha256",
    "prepared_sha256",
    "raw_characters",
    "prepared_characters",
    "unique_characters",
    "train_split",
    "train_characters",
    "validation_characters",
    "normalization_rules",
]


def create_run_dir(runs_dir: str | Path, run_name: str) -> Path:
    base = Path(runs_dir) / run_name
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    candidate = base / timestamp
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = base / f"{timestamp}_{suffix:03d}"
    candidate.mkdir(parents=True)
    return candidate


def write_config_snapshot(path: str | Path, config: ExperimentConfig) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    lines: list[str] = []
    for section, values in data.items():
        lines.append(f"{section}:")
        for key, value in values.items():
            lines.append(f"  {key}: {json.dumps(value, ensure_ascii=False, allow_nan=False)}")
        lines.append("")
    atomic_write_text(output, "\n".join(lines).rstrip() + "\n")


class MetricsWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"dataset manifest not found at {manifest_path}. "
            "Run scripts/prepare_corpus.py with --manifest before training."
        )
    return cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))


def copy_dataset_manifest(
    manifest_path: str | Path, run_dir: str | Path
) -> tuple[Path, dict[str, Any]]:
    manifest = load_dataset_manifest(manifest_path)
    destination = Path(run_dir) / "dataset_manifest.json"
    atomic_write_text(destination, Path(manifest_path).read_text(encoding="utf-8"))
    return destination, manifest


def verify_dataset_manifest(
    manifest: dict[str, Any],
    *,
    prepared_path: str | Path,
    prepared_text: str,
    train_split: float,
) -> None:
    required = {"prepared_sha256", "prepared_characters", "train_split"}
    missing = sorted(required - manifest.keys())
    if missing:
        raise ValueError(f"dataset manifest is missing required fields: {', '.join(missing)}")
    loaded_sha256 = hashlib.sha256(prepared_text.encode("utf-8")).hexdigest()
    if manifest["prepared_sha256"] != loaded_sha256:
        raise ValueError("prepared corpus checksum does not match dataset manifest")
    if int(manifest["prepared_characters"]) != len(prepared_text):
        raise ValueError("prepared corpus character count does not match dataset manifest")
    if float(manifest["train_split"]) != train_split:
        raise ValueError("configured train split does not match dataset manifest")
    split_index = int(len(prepared_text) * train_split)
    expected_counts = {
        "train_characters": split_index,
        "validation_characters": len(prepared_text) - split_index,
    }
    for field, expected in expected_counts.items():
        if field in manifest and int(manifest[field]) != expected:
            raise ValueError(f"dataset manifest {field} does not match configured split")


def dataset_summary_from_manifest(
    manifest: dict[str, Any],
    *,
    run_manifest_path: str | Path,
) -> dict[str, Any]:
    summary = {field: manifest.get(field) for field in DATASET_SUMMARY_FIELDS}
    summary["manifest_path"] = str(run_manifest_path)
    return summary
