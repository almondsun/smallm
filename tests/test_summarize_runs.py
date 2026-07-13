import json

import pytest

from scripts.summarize_runs import _observation
from smallm.config import ExperimentConfig, TrainConfig
from smallm.evaluation.robustness import summarize_observations
from smallm.training.artifacts import write_config_snapshot


def _write_run(tmp_path, name, *, seed=1, learning_rate=1e-3, summary_updates=None):
    run_dir = tmp_path / name
    run_dir.mkdir()
    write_config_snapshot(
        run_dir / "config.yaml",
        ExperimentConfig(
            train=TrainConfig(
                run_name=name,
                runs_dir=str(tmp_path),
                seed=seed,
                learning_rate=learning_rate,
            )
        ),
    )
    summary = {
        "schema_version": 2,
        "status": "complete",
        "actual_steps": 100,
        "best_val_step": 80,
        "best_val_bits_per_char": 2.0,
        "final_val_bits_per_char": 2.1,
        "duration_seconds": 5.0,
        "validation_mode": "full",
        "validation_coverage": 1.0,
        "tokenizer_type": "char",
        "tokenizer_vocab_size": 10,
        "dataset": {
            "prepared_sha256": "abc",
            "train_split": 0.9,
            "train_characters": 90,
            "validation_characters": 10,
        },
    }
    summary.update(summary_updates or {})
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return run_dir


def test_observations_verify_training_comparability(tmp_path):
    first = _observation(_write_run(tmp_path, "first", seed=1))
    second = _observation(_write_run(tmp_path, "second", seed=2))
    different = _observation(_write_run(tmp_path, "different", seed=3, learning_rate=2e-3))

    assert first.comparison_fingerprint == second.comparison_fingerprint
    summarize_observations([first, second])
    with pytest.raises(ValueError, match="fingerprint"):
        summarize_observations([first, different])


@pytest.mark.parametrize(
    "updates",
    [
        {"actual_steps": 12.5},
        {"actual_steps": -1},
        {"actual_steps": 10, "best_val_step": 11},
        {"best_val_bits_per_char": float("nan")},
        {"final_val_bits_per_char": float("inf")},
        {"duration_seconds": -1},
        {"actual_steps": 10**100},
        {"best_val_bits_per_char": 10**1000},
        {"final_val_bits_per_char": 10**1000},
        {"duration_seconds": 10**1000},
    ],
)
def test_observation_rejects_corrupt_numeric_fields(tmp_path, updates):
    run_dir = _write_run(tmp_path, "invalid", summary_updates=updates)

    with pytest.raises(ValueError):
        _observation(run_dir)


def test_observation_reports_malformed_artifact_path(tmp_path):
    run_dir = tmp_path / "malformed"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="summary"):
        _observation(run_dir)


def test_observation_rejects_oversized_summary_without_reading_it_all(tmp_path):
    run_dir = tmp_path / "oversized"
    run_dir.mkdir()
    (run_dir / "summary.json").write_bytes(b" " * 1_000_001)

    with pytest.raises(ValueError, match="exceeds 1000000 bytes"):
        _observation(run_dir)
