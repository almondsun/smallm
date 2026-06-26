import json

from smallm.config import ExperimentConfig, TrainConfig
from smallm.training.artifacts import MetricsWriter, create_run_dir, write_config_snapshot, write_json


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
