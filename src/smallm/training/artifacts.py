from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from smallm.config import ExperimentConfig


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
