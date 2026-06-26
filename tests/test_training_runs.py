import json

import pytest

from smallm.training.runs import (
    checkpoint_path_for_run,
    find_latest_run,
    list_all_run_dirs,
    list_run_dirs,
    list_run_names,
    load_last_metric,
    load_summary,
    resolve_run_path,
)


def _write_run(root, run_name, run_id, metric):
    run_dir = root / run_name / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint_path": str(run_dir / "checkpoint.pt"),
                "final_train_loss": metric["train_loss"],
                "final_val_loss": metric["val_loss"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(json.dumps(metric) + "\n", encoding="utf-8")
    return run_dir


def test_list_and_find_latest_run(tmp_path):
    first = _write_run(tmp_path, "smoke", "2026-01-01_00-00-00", {"step": 1, "train_loss": 3.0, "val_loss": None})
    latest = _write_run(tmp_path, "smoke", "2026-01-01_00-00-01", {"step": 2, "train_loss": 2.0, "val_loss": 1.9})

    assert list_run_names(tmp_path) == ["smoke"]
    assert list_run_dirs("smoke", tmp_path) == [first, latest]
    assert list_all_run_dirs(tmp_path) == [first, latest]
    assert find_latest_run("smoke", tmp_path) == latest
    assert resolve_run_path("latest", run_name="smoke", runs_dir=tmp_path) == latest


def test_load_summary_last_metric_and_checkpoint_path(tmp_path):
    run_dir = _write_run(tmp_path, "tiny_gpt", "run-0001", {"step": 5, "train_loss": 1.2, "val_loss": 1.1})

    assert load_summary(run_dir)["final_train_loss"] == 1.2
    assert load_last_metric(run_dir) == {"step": 5, "train_loss": 1.2, "val_loss": 1.1}
    assert checkpoint_path_for_run(run_dir) == run_dir / "checkpoint.pt"


def test_resolve_latest_requires_run_name(tmp_path):
    with pytest.raises(ValueError):
        resolve_run_path("latest", runs_dir=tmp_path)
