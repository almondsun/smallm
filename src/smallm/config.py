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
    prepared_path: str = "data/processed/corpus.txt"
    manifest_path: str = "data/processed/corpus_manifest.json"
    tokenizer_path: str = "data/processed/tokenizer.json"
    tokenizer_type: str = "char"
    bpe_vocab_size: int | None = None
    bpe_min_frequency: int = 2
    block_size: int = 128
    train_split: float = 0.9

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("data.block_size must be positive")
        if not 0.0 < self.train_split < 1.0:
            raise ValueError("data.train_split must be between 0 and 1")
        if self.tokenizer_type not in {"char", "bpe"}:
            raise ValueError("data.tokenizer_type must be 'char' or 'bpe'")
        if self.tokenizer_type == "bpe":
            if self.bpe_vocab_size is None or self.bpe_vocab_size <= 0:
                raise ValueError("data.bpe_vocab_size must be positive for BPE tokenizers")
            if self.bpe_min_frequency <= 0:
                raise ValueError("data.bpe_min_frequency must be positive")


@dataclass(frozen=True)
class ModelConfig:
    # Default for incomplete configs; training uses the tokenizer-derived vocab size.
    vocab_size: int = 256
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if self.n_embd % self.n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainConfig:
    run_name: str = "gptiny"
    runs_dir: str = "runs"
    batch_size: int = 32
    max_steps: int = 1000
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int | None = 5
    sample_prompt: str = "Once"
    sample_max_new_tokens: int = 100
    sample_temperature: float = 1.0
    sample_top_k: int | None = None
    sample_seed: int | None = None
    sample_greedy: bool = False
    seed: int = 1337

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ValueError("train.run_name must not be empty")
        for name in ("batch_size", "max_steps", "log_interval", "eval_interval"):
            if getattr(self, name) <= 0:
                raise ValueError(f"train.{name} must be positive")
        if self.eval_batches is not None and self.eval_batches <= 0:
            raise ValueError("train.eval_batches must be positive or null")
        if self.learning_rate <= 0:
            raise ValueError("train.learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("train.weight_decay must be non-negative")
        if not self.sample_prompt:
            raise ValueError("train.sample_prompt must not be empty")
        if self.sample_max_new_tokens < 0:
            raise ValueError("train.sample_max_new_tokens must be non-negative")
        if self.sample_temperature <= 0:
            raise ValueError("train.sample_temperature must be positive")
        if self.sample_top_k is not None and self.sample_top_k <= 0:
            raise ValueError("train.sample_top_k must be positive or null")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()

    def __post_init__(self) -> None:
        if self.data.block_size > self.model.block_size:
            raise ValueError("data.block_size cannot exceed model.block_size")


def _section(cls: type, values: dict[str, Any] | None) -> Any:
    return cls(**(values or {}))


def _coerce_scalar(value: str) -> Any:
    if value in {"null", "None"}:
        return None
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
