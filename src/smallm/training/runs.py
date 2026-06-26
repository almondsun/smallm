from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def list_run_names(runs_dir: str | Path = "runs") -> list[str]:
    root = Path(runs_dir)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def list_run_dirs(run_name: str, runs_dir: str | Path = "runs") -> list[Path]:
    root = Path(runs_dir) / run_name
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def list_all_run_dirs(runs_dir: str | Path = "runs") -> list[Path]:
    run_dirs: list[Path] = []
    for run_name in list_run_names(runs_dir):
        run_dirs.extend(list_run_dirs(run_name, runs_dir))
    return sorted(run_dirs)


def find_latest_run(run_name: str, runs_dir: str | Path = "runs") -> Path:
    run_dirs = list_run_dirs(run_name, runs_dir)
    if not run_dirs:
        raise FileNotFoundError(f"no runs found for {run_name!r} under {runs_dir}")
    return run_dirs[-1]


def resolve_run_path(
    run: str | Path,
    *,
    run_name: str | None = None,
    runs_dir: str | Path = "runs",
) -> Path:
    if str(run) == "latest":
        if run_name is None:
            raise ValueError("--run-name is required when --run latest is used")
        return find_latest_run(run_name, runs_dir)
    return Path(run)


def load_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_last_metric(run_dir: str | Path) -> dict[str, Any] | None:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return None
    last_line = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last_line = line
    if last_line is None:
        return None
    return json.loads(last_line)


def checkpoint_path_for_run(run_dir: str | Path) -> Path:
    return Path(run_dir) / "checkpoint.pt"
