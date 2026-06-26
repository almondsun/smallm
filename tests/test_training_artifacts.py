import json

import pytest

from smallm.config import ExperimentConfig, TrainConfig
from smallm.training.artifacts import (
    MetricsWriter,
    copy_dataset_manifest,
    create_run_dir,
    dataset_summary_from_manifest,
    load_dataset_manifest,
    write_config_snapshot,
    write_json,
)


def test_create_run_dir_creates_unique_run_directories(tmp_path):
    first = create_run_dir(tmp_path, "smoke")
    second = create_run_dir(tmp_path, "smoke")

    assert first.exists()
    assert second.exists()
    assert first != second
    assert first.parent == tmp_path / "smoke"


def test_metrics_writer_writes_jsonl(tmp_path):
    path = tmp_path / "metrics.jsonl"

    with MetricsWriter(path) as metrics:
        metrics.write({"step": 1, "train_loss": 3.0})

    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 1, "train_loss": 3.0}


def test_write_config_snapshot_and_summary_json(tmp_path):
    config_path = tmp_path / "config.yaml"
    summary_path = tmp_path / "summary.json"

    write_config_snapshot(config_path, ExperimentConfig(train=TrainConfig(run_name="smoke")))
    write_json(summary_path, {"ok": True})

    assert "run_name: smoke" in config_path.read_text(encoding="utf-8")
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {"ok": True}


def test_dataset_manifest_copy_and_summary_fields(tmp_path):
    manifest_path = tmp_path / "corpus_manifest.json"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = {
        "source_name": "source",
        "source_note": "note",
        "raw_sha256": "raw-hash",
        "prepared_sha256": "prepared-hash",
        "raw_characters": 10,
        "prepared_characters": 8,
        "unique_characters": 4,
        "train_split": 0.75,
        "train_characters": 6,
        "validation_characters": 2,
        "normalization_rules": ["rule"],
        "extra": "ignored",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    copied_path, loaded = copy_dataset_manifest(manifest_path, run_dir)
    summary = dataset_summary_from_manifest(loaded, run_manifest_path=copied_path)

    assert copied_path == run_dir / "dataset_manifest.json"
    assert json.loads(copied_path.read_text(encoding="utf-8")) == manifest
    assert summary == {
        "source_name": "source",
        "source_note": "note",
        "raw_sha256": "raw-hash",
        "prepared_sha256": "prepared-hash",
        "raw_characters": 10,
        "prepared_characters": 8,
        "unique_characters": 4,
        "train_split": 0.75,
        "train_characters": 6,
        "validation_characters": 2,
        "normalization_rules": ["rule"],
        "manifest_path": str(copied_path),
    }


def test_missing_dataset_manifest_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run scripts/prepare_corpus.py with --manifest"):
        load_dataset_manifest(tmp_path / "missing.json")
