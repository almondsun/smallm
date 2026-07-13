import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.summarize_matrix import analyze_matrix
from smallm.config import DataConfig, ExperimentConfig, TrainConfig
from smallm.training.artifacts import write_config_snapshot

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _write_run(tmp_path, corpus, tokenizer, seed, *, bpc=2.0, checksum=None):
    run_dir = tmp_path / f"{corpus}-{tokenizer}-{seed}"
    run_dir.mkdir()
    write_config_snapshot(
        run_dir / "config.yaml",
        ExperimentConfig(
            data=DataConfig(
                tokenizer_type=tokenizer,
                bpe_vocab_size=512 if tokenizer == "byte_bpe" else None,
            ),
            train=TrainConfig(run_name=run_dir.name, runs_dir=str(tmp_path), seed=seed),
        ),
    )
    summary = {
        "schema_version": 2,
        "status": "complete",
        "actual_steps": 100,
        "best_val_step": 80,
        "best_val_bits_per_char": bpc,
        "final_val_bits_per_char": bpc + 0.1,
        "duration_seconds": 5.0,
        "validation_mode": "full",
        "validation_coverage": 1.0,
        "tokenizer_type": tokenizer,
        "tokenizer_vocab_size": 10,
        "dataset": {
            "prepared_sha256": checksum or ("a" * 64 if corpus == "alice" else "b" * 64),
            "train_split": 0.9,
            "train_characters": 90,
            "validation_characters": 10,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return (corpus, tokenizer, run_dir)


def _matrix(tmp_path):
    return [
        _write_run(
            tmp_path,
            corpus,
            tokenizer,
            seed,
            bpc=2.0
            + (0.1 if corpus == "peter_pan" else 0.0)
            - (0.02 if tokenizer == "byte_bpe" else 0.0),
        )
        for corpus in ("alice", "peter_pan")
        for tokenizer in ("char", "byte_bpe")
        for seed in (1, 2, 3)
    ]


def test_analyze_matrix_computes_paired_contrasts_and_interaction(tmp_path):
    analysis = analyze_matrix(
        _matrix(tmp_path), reference_tokenizer="char", candidate_tokenizer="byte_bpe"
    )

    assert [seed for seed, _ in analysis.contrasts["alice"]] == [1, 2, 3]
    assert [value for _, value in analysis.contrasts["alice"]] == pytest.approx([-0.02] * 3)
    assert [value for _, value in analysis.contrasts["peter_pan"]] == pytest.approx([-0.02] * 3)
    assert [value for _, value in analysis.interactions] == pytest.approx([0.0] * 3)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda entries: [
                entry
                for entry in entries
                if not (entry[0] == "peter_pan" and entry[1] == "byte_bpe")
            ],
            "missing",
        ),
        (lambda entries: [*entries, entries[0]], "distinct"),
        (
            lambda entries: [("Bad Label", tokenizer, path) for _, tokenizer, path in entries],
            "labels",
        ),
        (
            lambda entries: [
                (corpus, tokenizer, path)
                for corpus, tokenizer, path in entries
                if not (corpus == "alice" and tokenizer == "char" and path.name.endswith("-3"))
            ],
            "seed set",
        ),
    ],
)
def test_analyze_matrix_rejects_unbalanced_design(tmp_path, mutation, match):
    with pytest.raises(ValueError, match=match):
        analyze_matrix(
            mutation(_matrix(tmp_path)),
            reference_tokenizer="char",
            candidate_tokenizer="byte_bpe",
        )


def test_analyze_matrix_rejects_mislabeled_tokenizer(tmp_path):
    entries = _matrix(tmp_path)
    corpus, _, path = entries[0]
    entries[0] = (corpus, "wrong", path)

    with pytest.raises(ValueError, match="tokenizer label"):
        analyze_matrix(entries, reference_tokenizer="char", candidate_tokenizer="byte_bpe")


def test_analyze_matrix_rejects_corpus_aliasing(tmp_path):
    entries = _matrix(tmp_path)
    for corpus, _, path in entries:
        if corpus == "peter_pan":
            payload = json.loads((path / "summary.json").read_text(encoding="utf-8"))
            payload["dataset"]["prepared_sha256"] = "a" * 64
            (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="different corpus labels"):
        analyze_matrix(entries, reference_tokenizer="char", candidate_tokenizer="byte_bpe")


def test_summarize_matrix_supports_direct_script_execution():
    completed = subprocess.run(
        [sys.executable, "scripts/summarize_matrix.py", "--help"],
        cwd=_REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--reference-tokenizer" in completed.stdout
