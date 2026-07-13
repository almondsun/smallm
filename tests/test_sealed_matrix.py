import json
from dataclasses import replace

import pytest

from smallm.config import DataConfig, ExperimentConfig, TrainConfig
from smallm.evaluation.sealed_matrix import (
    SealedTestObservation,
    analyze_sealed_test_matrix,
    load_sealed_test_observation,
)
from smallm.training.artifacts import write_config_snapshot


def _observation(seed, tokenizer, corpus, fingerprint, bpc):
    return SealedTestObservation(
        seed=seed,
        tokenizer=tokenizer,
        prepared_sha256=corpus,
        comparison_fingerprint=fingerprint,
        checkpoint_step=10,
        checkpoint_sha256="c" * 64,
        test_loss=2.0,
        test_bpc=bpc,
        test_tokens=100,
        target_tokens=99,
        target_characters=99,
    )


def test_sealed_matrix_computes_paired_contrasts(monkeypatch, tmp_path):
    observations = {}
    entries = []
    for corpus_index, corpus in enumerate(("first", "second")):
        for tokenizer in ("char", "byte_bpe"):
            for seed in (1, 2):
                path = tmp_path / f"{corpus}-{tokenizer}-{seed}"
                path.mkdir()
                entries.append((corpus, tokenizer, path))
                baseline = 2.0 + corpus_index * 0.1 + seed * 0.01
                bpc = baseline if tokenizer == "char" else baseline - 0.05
                observations[path] = _observation(
                    seed,
                    tokenizer,
                    str(corpus_index) * 64,
                    f"{corpus}-{tokenizer}",
                    bpc,
                )
    monkeypatch.setattr(
        "smallm.evaluation.sealed_matrix.load_sealed_test_observation",
        lambda path: observations[path],
    )

    analysis = analyze_sealed_test_matrix(
        entries, reference_tokenizer="char", candidate_tokenizer="byte_bpe"
    )

    assert [seed for seed, _ in analysis.contrasts["first"]] == [1, 2]
    assert [value for _, value in analysis.contrasts["first"]] == pytest.approx([-0.05, -0.05])
    assert [value for _, value in analysis.contrasts["second"]] == pytest.approx([-0.05, -0.05])
    assert [value for _, value in analysis.interactions] == pytest.approx([0.0, 0.0])


def test_sealed_matrix_rejects_unbalanced_or_inconsistent_cells(monkeypatch, tmp_path):
    paths = [tmp_path / str(index) for index in range(8)]
    for path in paths:
        path.mkdir()
    observations = {
        path: _observation(
            1 + index % 2,
            "char" if (index // 2) % 2 == 0 else "byte_bpe",
            "a" * 64 if index < 4 else "b" * 64,
            f"fingerprint-{index // 2}",
            2.0,
        )
        for index, path in enumerate(paths)
    }
    monkeypatch.setattr(
        "smallm.evaluation.sealed_matrix.load_sealed_test_observation",
        lambda path: observations[path],
    )
    entries = [
        ("first" if index < 4 else "second", observations[path].tokenizer, path)
        for index, path in enumerate(paths)
    ]
    analyze_sealed_test_matrix(entries, reference_tokenizer="char", candidate_tokenizer="byte_bpe")

    observations[paths[-1]] = replace(observations[paths[-1]], seed=3)
    with pytest.raises(ValueError, match="same seed set"):
        analyze_sealed_test_matrix(
            entries, reference_tokenizer="char", candidate_tokenizer="byte_bpe"
        )


def test_load_sealed_test_observation_validates_run_identity(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = ExperimentConfig(
        data=DataConfig(train_split=0.8, validation_split=0.1),
        train=TrainConfig(seed=2027, run_name="fixture", max_steps=10),
    )
    write_config_snapshot(run_dir / "config.yaml", config)
    prepared_sha256 = "a" * 64
    summary = {
        "schema_version": 2,
        "status": "complete",
        "actual_steps": 10,
        "best_val_step": 8,
        "best_val_bits_per_char": 2.0,
        "final_val_bits_per_char": 2.1,
        "duration_seconds": 1.0,
        "tokenizer_type": "char",
        "validation_mode": "full",
        "validation_coverage": 1.0,
        "test_status": "sealed_unread",
        "test_characters": 100,
        "dataset": {
            "prepared_sha256": prepared_sha256,
            "train_split": 0.9,
            "train_characters": 800,
            "validation_characters": 100,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    test_result = {
        "schema_version": 1,
        "status": "complete",
        "checkpoint_kind": "best",
        "checkpoint_step": 8,
        "checkpoint_sha256": "b" * 64,
        "prepared_sha256": prepared_sha256,
        "evaluation_mode": "full",
        "test_coverage": 1.0,
        "test_characters": 100,
        "test_tokens": 100,
        "test_target_tokens": 99,
        "test_total_target_tokens": 99,
        "test_target_characters": 99,
        "test_loss": 1.5,
        "test_bits_per_character": 2.2,
    }
    path = run_dir / "test_evaluation_best.json"
    path.write_text(json.dumps(test_result), encoding="utf-8")

    observation = load_sealed_test_observation(run_dir)
    assert observation.seed == 2027
    assert observation.test_bpc == 2.2
    assert observation.checkpoint_step == 8

    test_result["prepared_sha256"] = "c" * 64
    path.write_text(json.dumps(test_result), encoding="utf-8")
    with pytest.raises(ValueError, match="corpus checksum"):
        load_sealed_test_observation(run_dir)
