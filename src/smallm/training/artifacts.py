from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
from typing import Any

from smallm.config import ExperimentConfig

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
            lines.append(f"  {key}: {value}")
        lines.append("")
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


class MetricsWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "MetricsWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_dataset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"dataset manifest not found at {manifest_path}. "
            "Run scripts/prepare_corpus.py with --manifest before training."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def copy_dataset_manifest(manifest_path: str | Path, run_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    manifest = load_dataset_manifest(manifest_path)
    destination = Path(run_dir) / "dataset_manifest.json"
    shutil.copyfile(manifest_path, destination)
    return destination, manifest


def dataset_summary_from_manifest(
    manifest: dict[str, Any],
    *,
    run_manifest_path: str | Path,
) -> dict[str, Any]:
    summary = {field: manifest.get(field) for field in DATASET_SUMMARY_FIELDS}
    summary["manifest_path"] = str(run_manifest_path)
    return summary
