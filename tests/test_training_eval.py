import json

import torch
from torch.utils.data import DataLoader

from smallm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from smallm.data import TokenBlockDataset
from smallm.model import GPT, GPTConfig
from smallm.training.trainer import _build_optimizer, _update_best_validation, estimate_loss, train


def test_estimate_loss_returns_loss_and_restores_train_mode():
    model = GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1, n_head=1, n_embd=8))
    model.train()
    dataset = TokenBlockDataset([0, 1, 2, 3, 4, 5, 6, 7], block_size=4)
    loader = DataLoader(dataset, batch_size=2)

    loss = estimate_loss(model, loader, torch.device("cpu"), max_batches=1)

    assert loss is not None
    assert loss > 0
    assert model.training


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
    assert summary["best_val_step"] in {2, 3}
    assert metrics[-1]["step"] == 3
    assert metrics[-1]["val_loss"] == summary["final_val_loss"]
