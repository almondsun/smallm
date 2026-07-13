import json

import pytest

from smallm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from smallm.data.corpus import file_sha256
from smallm.evaluation import evaluate_sealed_test
from smallm.training.trainer import train


def test_sealed_test_is_unscored_during_training_and_evaluated_once(tmp_path):
    prepared_path = tmp_path / "corpus.txt"
    manifest_path = tmp_path / "corpus_manifest.json"
    text = "Once upon a time.\n" * 20
    prepared_path.write_text(text, encoding="utf-8")
    train_end = int(len(text) * 0.6)
    validation_end = int(len(text) * 0.8)
    manifest_path.write_text(
        json.dumps(
            {
                "source_name": "sealed fixture",
                "prepared_sha256": file_sha256(prepared_path),
                "prepared_characters": len(text),
                "train_split": 0.6,
                "validation_split": 0.2,
                "train_characters": train_end,
                "validation_characters": validation_end - train_end,
                "test_characters": len(text) - validation_end,
            }
        ),
        encoding="utf-8",
    )
    config = ExperimentConfig(
        data=DataConfig(
            prepared_path=str(prepared_path),
            manifest_path=str(manifest_path),
            tokenizer_path=str(tmp_path / "tokenizer.json"),
            block_size=4,
            train_split=0.6,
            validation_split=0.2,
        ),
        model=ModelConfig(vocab_size=256, block_size=4, n_layer=1, n_head=1, n_embd=8),
        train=TrainConfig(
            run_name="sealed",
            runs_dir=str(tmp_path / "runs"),
            batch_size=2,
            max_steps=2,
            log_interval=1,
            eval_interval=1,
            eval_batches=None,
            sample_max_new_tokens=1,
            seed=1337,
            sample_seed=1337,
        ),
    )

    run_dir = train(config).parent
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["test_status"] == "sealed_unread"
    assert summary["test_characters"] == len(text) - validation_end
    assert "test_tokens" not in summary
    assert "test_loss" not in summary

    summary_path = run_dir / "summary.json"
    invalid_summary = {**summary, "test_characters": summary["test_characters"] + 1}
    summary_path.write_text(json.dumps(invalid_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the sealed test split"):
        evaluate_sealed_test(run_dir)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    output_path, evaluation = evaluate_sealed_test(run_dir)
    assert output_path == run_dir / "test_evaluation_best.json"
    assert evaluation["checkpoint_kind"] == "best"
    assert evaluation["checkpoint_step"] == summary["best_val_step"]
    assert evaluation["test_characters"] == len(text) - validation_end
    assert evaluation["test_coverage"] == 1.0
    assert evaluation["evaluation_mode"] == "full"
    assert evaluation["test_bits_per_character"] > 0

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_sealed_test(run_dir)
