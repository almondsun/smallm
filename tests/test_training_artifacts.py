import json

import pytest
import torch

from smallm.config import ExperimentConfig, TrainConfig
from smallm.data.corpus import file_sha256
from smallm.training.artifacts import (
    MetricsWriter,
    copy_dataset_manifest,
    create_run_dir,
    dataset_summary_from_manifest,
    load_dataset_manifest,
    verify_dataset_manifest,
    write_config_snapshot,
    write_json,
)
from smallm.training.checkpoints import load_checkpoint, save_checkpoint


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


def test_artifact_json_rejects_non_finite_numbers(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    with MetricsWriter(metrics_path) as metrics, pytest.raises(ValueError):
        metrics.write({"train_loss": float("nan")})
    assert metrics_path.read_text(encoding="utf-8") == ""

    summary_path = tmp_path / "summary.json"
    with pytest.raises(ValueError):
        write_json(summary_path, {"loss": float("inf")})
    assert not summary_path.exists()


def test_write_config_snapshot_and_summary_json(tmp_path):
    config_path = tmp_path / "config.yaml"
    summary_path = tmp_path / "summary.json"

    write_config_snapshot(config_path, ExperimentConfig(train=TrainConfig(run_name="smoke")))
    write_json(summary_path, {"ok": True})

    assert 'run_name: "smoke"' in config_path.read_text(encoding="utf-8")
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


def test_verify_dataset_manifest_rejects_stale_metadata(tmp_path):
    prepared = tmp_path / "corpus.txt"
    text = "abcdefghij"
    prepared.write_text(text, encoding="utf-8")
    valid = {
        "prepared_sha256": file_sha256(prepared),
        "prepared_characters": 10,
        "train_split": 0.8,
        "train_characters": 8,
        "validation_characters": 2,
    }
    verify_dataset_manifest(valid, prepared_path=prepared, prepared_text=text, train_split=0.8)

    variants = [
        {},
        {**valid, "prepared_sha256": "wrong"},
        {**valid, "prepared_characters": 9},
        {**valid, "train_split": 0.9},
        {**valid, "train_characters": 7},
        {**valid, "validation_characters": 3},
    ]
    for manifest in variants:
        with pytest.raises(ValueError):
            verify_dataset_manifest(
                manifest,
                prepared_path=prepared,
                prepared_text=text,
                train_split=0.8,
            )


def test_checkpoint_round_trip_and_schema_validation(tmp_path):
    path = tmp_path / "checkpoint.pt"
    payload = {
        "model_state": {"weight": torch.tensor([1.0])},
        "model_config": {
            "vocab_size": 8,
            "block_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "dropout": 0.0,
        },
        "tokenizer_path": "tokenizer.json",
    }
    save_checkpoint(path, payload)

    assert load_checkpoint(path)["model_config"]["vocab_size"] == 8

    for invalid in [[], {}, {"model_state": [], "model_config": {}}]:
        torch.save(invalid, path)
        with pytest.raises(ValueError):
            load_checkpoint(path)


def test_checkpoint_rejects_unsupported_schema_and_config(tmp_path):
    path = tmp_path / "checkpoint.pt"
    base = {
        "model_state": {"weight": torch.tensor([1.0])},
        "model_config": {
            "vocab_size": 8,
            "block_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "dropout": 0.0,
        },
        "tokenizer": {"type": "char", "stoi": {"a": 0}},
        "step": 1,
    }
    for payload in [
        {**base, "schema_version": 99},
        {**base, "schema_version": 2, "step": -1},
        {**base, "schema_version": 2, "tokenizer": []},
        {**base, "schema_version": 2, "model_state": {"weight": "not a tensor"}},
        {**base, "schema_version": 1, "tokenizer": None},
        {**base, "schema_version": 2, "model_config": {}},
        {
            **base,
            "schema_version": 2,
            "model_config": {**base["model_config"], "vocab_size": 1_000_001},
        },
        {
            **base,
            "schema_version": 2,
            "model_config": {**base["model_config"], "dropout": 1.0},
        },
        {**base, "schema_version": 2, "model_config": {**base["model_config"], "n_head": 3}},
    ]:
        torch.save(payload, path)
        with pytest.raises(ValueError):
            load_checkpoint(path)


def test_checkpoint_validates_v2_tokenizer_and_parameter_budget(tmp_path):
    path = tmp_path / "checkpoint.pt"
    config = {
        "vocab_size": 2,
        "block_size": 4,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "dropout": 0.0,
    }
    base = {
        "schema_version": 2,
        "model_state": {"weight": torch.tensor([1.0])},
        "model_config": config,
        "tokenizer": {
            "schema_version": 2,
            "type": "char",
            "stoi": {"a": 0, "<unk>": 1},
            "unk_token": "<unk>",
            "vocab_size": 2,
        },
        "step": 1,
    }
    torch.save(base, path)
    assert load_checkpoint(path)["schema_version"] == 2

    invalid_states = [
        {**base["tokenizer"], "stoi": {"a": 0, "b": 2}},
        {**base["tokenizer"], "vocab_size": 3},
        {**base["tokenizer"], "schema_version": 99},
    ]
    for tokenizer in invalid_states:
        torch.save({**base, "tokenizer": tokenizer}, path)
        with pytest.raises(ValueError):
            load_checkpoint(path)

    legacy_invalid = {
        **base,
        "schema_version": 1,
        "tokenizer": {"type": "char", "stoi": {"a": 0, "b": 2}},
    }
    torch.save(legacy_invalid, path)
    with pytest.raises(ValueError, match="contiguous"):
        load_checkpoint(path)

    huge = {**config, "vocab_size": 100_000, "n_embd": 8_192, "n_head": 128}
    torch.save({**base, "model_config": huge}, path)
    with pytest.raises(ValueError, match="parameter budget"):
        load_checkpoint(path)


def test_checkpoint_validates_bpe_merge_structure(tmp_path):
    path = tmp_path / "checkpoint.pt"
    config = {
        "vocab_size": 4,
        "block_size": 4,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "dropout": 0.0,
    }
    tokenizer = {
        "schema_version": 2,
        "type": "bpe",
        "vocab": {"<unk>": 0, "a": 1, "b": 2, "ab": 3},
        "unk_token": "<unk>",
        "vocab_size": 4,
        "merges": [["a", "b"]],
    }
    base = {
        "schema_version": 2,
        "model_state": {"weight": torch.tensor([1.0])},
        "model_config": config,
        "tokenizer": tokenizer,
        "step": 1,
    }
    torch.save(base, path)
    assert load_checkpoint(path)["tokenizer"]["type"] == "bpe"

    invalid_states = [
        {**tokenizer, "unk_token": "missing"},
        {**tokenizer, "merges": "not a list"},
        {**tokenizer, "merges": [["a", "missing"]]},
        {**tokenizer, "vocab": {"<unk>": 0, "a": 1, "b": 2, "other": 3}},
    ]
    for state in invalid_states:
        torch.save({**base, "tokenizer": state}, path)
        with pytest.raises(ValueError):
            load_checkpoint(path)


def test_checkpoint_validates_byte_bpe_structure(tmp_path):
    path = tmp_path / "checkpoint.pt"
    vocab = {f"{value:02x}": value for value in range(256)}
    vocab["0001"] = 256
    tokenizer = {
        "schema_version": 2,
        "type": "byte_bpe",
        "vocab": vocab,
        "vocab_size": 257,
        "merges": [["00", "01"]],
        "min_frequency": 2,
        "boundary_policy": "whitespace_segments",
        "encoding": "utf-8",
    }
    base = {
        "schema_version": 2,
        "model_state": {"weight": torch.tensor([1.0])},
        "model_config": {
            "vocab_size": 257,
            "block_size": 4,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "dropout": 0.0,
        },
        "tokenizer": tokenizer,
        "step": 1,
    }
    torch.save(base, path)
    assert load_checkpoint(path)["tokenizer"]["type"] == "byte_bpe"

    invalid_vocabs = []
    for replacement in ("zz", "0", "0002"):
        invalid_vocab = dict(vocab)
        del invalid_vocab["0001"]
        invalid_vocab[replacement] = 256
        invalid_vocabs.append(invalid_vocab)
    missing_base = dict(vocab)
    del missing_base["ff"]
    missing_base["0002"] = 255
    invalid_vocabs.append(missing_base)

    invalid_states = [
        *[{**tokenizer, "vocab": invalid_vocab} for invalid_vocab in invalid_vocabs],
        {**tokenizer, "encoding": "utf-16"},
        {**tokenizer, "boundary_policy": "none"},
        {**tokenizer, "min_frequency": True},
        {**tokenizer, "min_frequency": 0},
        {**tokenizer, "merges": [["00", "ff"]]},
    ]
    for state in invalid_states:
        torch.save({**base, "tokenizer": state}, path)
        with pytest.raises(ValueError):
            load_checkpoint(path)
