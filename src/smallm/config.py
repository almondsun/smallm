from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is absent.
    yaml = None


@dataclass(frozen=True)
class DataConfig:
    input_path: str = "data/raw/input.txt"
    tokenizer_path: str = "data/processed/tokenizer.json"
    block_size: int = 128
    train_split: float = 0.9


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 1000
    learning_rate: float = 3e-4
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 5
    checkpoint_dir: str = "checkpoints"
    seed: int = 1337


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def _section(cls: type, values: dict[str, Any] | None) -> Any:
    return cls(**(values or {}))


def _coerce_scalar(value: str) -> Any:
    if value in {"true", "false"}:
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not raw_line.startswith(" "):
            key = line.removesuffix(":")
            parsed[key] = {}
            current_section = parsed[key]
            continue
        if current_section is None:
            raise ValueError(f"invalid config line before section: {raw_line}")
        key, value = line.strip().split(":", 1)
        current_section[key] = _coerce_scalar(value.strip())
    return parsed


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if yaml is None:
        raw = _load_simple_yaml(config_path)
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    return ExperimentConfig(
        data=_section(DataConfig, raw.get("data")),
        model=_section(ModelConfig, raw.get("model")),
        train=_section(TrainConfig, raw.get("train")),
    )
