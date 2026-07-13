import json

import torch
from torch.utils.data import DataLoader

from smallm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from smallm.data import ByteBPETokenizer, CharTokenizer, TokenBlockDataset
from smallm.data.corpus import file_sha256
from smallm.model import GPT, GPTConfig
from smallm.training import load_checkpoint
from smallm.training.trainer import (
    _build_optimizer,
    _update_best_validation,
    _validation_starts,
    estimate_loss,
    evaluate_tokens,
    train,
)


def test_estimate_loss_returns_loss_and_restores_train_mode():
    model = GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=8))
    model.train()
    dataset = TokenBlockDataset([0, 1, 2, 3, 4, 5, 6, 7], block_size=4)
    loader = DataLoader(dataset, batch_size=2)

    loss = estimate_loss(model, loader, torch.device("cpu"), max_batches=1)

    assert loss is not None
    assert loss > 0
    assert model.training


def test_validation_blocks_are_non_overlapping_and_evenly_sampled():
    assert _validation_starts(11, 4, None) == [0, 4, 8]
    assert _validation_starts(21, 4, 3) == [0, 8, 16]
    assert _validation_starts(21, 4, 1) == [0]


def test_evaluate_tokens_reports_exact_full_coverage():
    tokenizer = CharTokenizer.train("abcdefgh")
    tokens = torch.tensor(tokenizer.encode("abcdefgh"))
    model = GPT(
        GPTConfig(vocab_size=tokenizer.vocab_size, block_size=3, n_layer=1, n_head=1, n_embd=8)
    )
    model.train()

    result = evaluate_tokens(
        model,
        tokens,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        block_size=3,
        max_batches=None,
    )

    assert result is not None
    assert result.mode == "full"
    assert result.target_tokens == result.total_target_tokens == 7
    assert result.target_characters == 7
    assert result.coverage == 1.0
    assert result.bits_per_character > 0
    assert model.training


def test_evaluate_tokens_uses_exact_unicode_character_counts():
    tokenizer = ByteBPETokenizer.train("é🙂 text", vocab_size=256)
    token_ids, character_counts = tokenizer.encode_with_character_counts("é🙂")
    tokens = torch.tensor(token_ids)
    model = GPT(
        GPTConfig(vocab_size=tokenizer.vocab_size, block_size=4, n_layer=1, n_head=1, n_embd=8)
    )

    result = evaluate_tokens(
        model,
        tokens,
        tokenizer=tokenizer,
        device=torch.device("cpu"),
        block_size=4,
        max_batches=None,
        character_counts=torch.tensor(character_counts),
    )

    assert result is not None
    assert result.target_characters == 1


def test_build_optimizer_uses_configured_weight_decay():
    model = GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=8))
    config = ExperimentConfig(train=TrainConfig(learning_rate=1e-4, weight_decay=0.01))

    optimizer = _build_optimizer(model, config)

    assert optimizer.param_groups[0]["lr"] == 1e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.01


def test_update_best_validation_tracks_lower_loss_only():
    best_loss, best_step = _update_best_validation(
        best_loss=None,
        best_step=None,
        val_loss=2.0,
        step=10,
    )
    best_loss, best_step = _update_best_validation(
        best_loss=best_loss,
        best_step=best_step,
        val_loss=2.5,
        step=20,
    )
    best_loss, best_step = _update_best_validation(
        best_loss=best_loss,
        best_step=best_step,
        val_loss=1.8,
        step=30,
    )

    assert best_loss == 1.8
    assert best_step == 30


def test_train_records_final_validation_when_max_steps_misses_eval_interval(tmp_path):
    prepared_path = tmp_path / "corpus.txt"
    manifest_path = tmp_path / "corpus_manifest.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    runs_dir = tmp_path / "runs"
    text = "Once upon a time\n" * 8
    prepared_path.write_text(text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "source_name": "test corpus",
                "prepared_sha256": file_sha256(prepared_path),
                "prepared_characters": len(text),
                "unique_characters": len(set(text)),
                "train_split": 0.8,
            }
        ),
        encoding="utf-8",
    )
    config = ExperimentConfig(
        data=DataConfig(
            prepared_path=str(prepared_path),
            manifest_path=str(manifest_path),
            tokenizer_path=str(tokenizer_path),
            block_size=4,
            train_split=0.8,
        ),
        model=ModelConfig(vocab_size=256, block_size=4, n_layer=1, n_head=1, n_embd=8),
        train=TrainConfig(
            run_name="final_eval",
            runs_dir=str(runs_dir),
            batch_size=2,
            max_steps=3,
            log_interval=10,
            eval_interval=2,
            eval_batches=1,
            sample_max_new_tokens=2,
            seed=1337,
            sample_seed=1337,
        ),
    )

    checkpoint_path = train(config)
    run_dir = checkpoint_path.parent
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    metrics = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["final_val_loss"] is not None
    assert summary["best_val_loss"] is not None
    validation_steps = {record["step"] for record in metrics if record["val_loss"] is not None}
    assert summary["best_val_step"] in validation_steps
    assert summary["best_checkpoint_path"] == str(run_dir / "best_checkpoint.pt")
    assert summary["best_checkpoint_exists"] is True
    assert (run_dir / "checkpoint.pt").exists()
    assert (run_dir / "best_checkpoint.pt").exists()
    assert load_checkpoint(run_dir / "best_checkpoint.pt")["step"] == summary["best_val_step"]
    assert metrics[-1]["step"] == 3
    assert metrics[-1]["val_loss"] == summary["final_val_loss"]


def test_train_can_use_bpe_tokenizer(tmp_path):
    prepared_path = tmp_path / "corpus.txt"
    manifest_path = tmp_path / "corpus_manifest.json"
    tokenizer_path = tmp_path / "tokenizer_bpe.json"
    runs_dir = tmp_path / "runs"
    text = ("abcdefghijklmnopqrstuvwxyz\n" * 8) + "Once more tokens\n"
    prepared_path.write_text(text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "source_name": "test corpus",
                "prepared_sha256": file_sha256(prepared_path),
                "prepared_characters": len(text),
                "unique_characters": len(set(text)),
                "train_split": 0.8,
            }
        ),
        encoding="utf-8",
    )
    config = ExperimentConfig(
        data=DataConfig(
            prepared_path=str(prepared_path),
            manifest_path=str(manifest_path),
            tokenizer_path=str(tokenizer_path),
            tokenizer_type="bpe",
            bpe_vocab_size=32,
            block_size=4,
            train_split=0.8,
        ),
        model=ModelConfig(vocab_size=256, block_size=4, n_layer=1, n_head=1, n_embd=8),
        train=TrainConfig(
            run_name="bpe_train",
            runs_dir=str(runs_dir),
            batch_size=2,
            max_steps=2,
            log_interval=1,
            eval_interval=1,
            eval_batches=1,
            sample_prompt="Once",
            sample_max_new_tokens=2,
            seed=1337,
            sample_seed=1337,
        ),
    )

    checkpoint_path = train(config)
    summary = json.loads((checkpoint_path.parent / "summary.json").read_text(encoding="utf-8"))

    assert summary["tokenizer_type"] == "bpe"
    assert summary["tokenizer_vocab_size"] <= 32
    assert summary["train_tokens"] > 0
    assert summary["val_tokens"] > 0
    assert summary["train_characters"] == int(len(text) * 0.8)
    assert summary["val_characters"] == len(text) - int(len(text) * 0.8)
    assert summary["final_val_bits_per_char"] is not None


def test_train_without_validation_does_not_create_best_checkpoint(tmp_path):
    prepared_path = tmp_path / "corpus.txt"
    manifest_path = tmp_path / "corpus_manifest.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    text = "Once upon a time\n" * 2
    prepared_path.write_text(text, encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "source_name": "test corpus",
                "prepared_sha256": file_sha256(prepared_path),
                "prepared_characters": len(text),
                "train_split": 0.99,
            }
        ),
        encoding="utf-8",
    )
    config = ExperimentConfig(
        data=DataConfig(
            prepared_path=str(prepared_path),
            manifest_path=str(manifest_path),
            tokenizer_path=str(tokenizer_path),
            block_size=4,
            train_split=0.99,
        ),
        model=ModelConfig(vocab_size=256, block_size=4, n_layer=1, n_head=1, n_embd=8),
        train=TrainConfig(
            run_name="no_validation",
            runs_dir=str(tmp_path / "runs"),
            batch_size=2,
            max_steps=1,
            log_interval=1,
            eval_interval=1,
            sample_prompt="Once",
            sample_max_new_tokens=1,
        ),
    )

    checkpoint_path = train(config)
    summary = json.loads((checkpoint_path.parent / "summary.json").read_text(encoding="utf-8"))

    assert checkpoint_path.exists()
    assert not (checkpoint_path.parent / "best_checkpoint.pt").exists()
    assert summary["best_checkpoint_exists"] is False
    assert summary["best_val_loss"] is None
    assert summary["best_val_step"] is None
