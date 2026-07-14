import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from smallm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from smallm.evaluation.capacity_panel import (
    CapacityPanelObservation,
    analyze_capacity_panel,
    load_capacity_panel_observation,
)
from smallm.evaluation.sealed_matrix import SealedTestObservation

ROOT = Path(__file__).resolve().parents[1]


def _panel(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    entries = []
    observations = {}
    specs = {
        "char128": ("char", None, 64, 128, 16, 826_000, 0.0),
        "char136": ("char", None, 64, 136, 16, 930_000, -0.02),
        "bytebpe512": ("byte_bpe", 512, 37, 128, 27, 929_664, -0.08),
    }
    for corpus_index, corpus in enumerate(("frankenstein", "douglass", "origin")):
        checksum = str(corpus_index + 1) * 64
        for arm, (tokenizer, vocab, block, width, batch, parameters, effect) in specs.items():
            config = ExperimentConfig(
                data=DataConfig(
                    tokenizer_type=tokenizer,
                    bpe_vocab_size=vocab,
                    block_size=block,
                    train_split=0.8,
                    validation_split=0.1,
                ),
                model=ModelConfig(block_size=block, n_layer=4, n_head=4, n_embd=width, dropout=0.1),
                train=TrainConfig(
                    run_name=f"{corpus}-{arm}",
                    batch_size=batch,
                    max_steps=5000,
                    learning_rate=0.001,
                    eval_interval=250,
                    eval_batches=None,
                    early_stopping_patience=3,
                ),
            )
            for seed in (1337, 2027, 4242):
                path = tmp_path / f"{corpus}-{arm}-{seed}"
                path.mkdir()
                entries.append((corpus, arm, path))
                observations[path] = CapacityPanelObservation(
                    sealed=SealedTestObservation(
                        seed=seed,
                        tokenizer=tokenizer,
                        prepared_sha256=checksum,
                        comparison_fingerprint=f"{corpus}-{arm}",
                        checkpoint_step=4000,
                        checkpoint_sha256="a" * 64,
                        test_loss=2.0,
                        test_bpc=2.2 + corpus_index * 0.1 + effect,
                        test_tokens=100,
                        target_tokens=99,
                        target_characters=99,
                    ),
                    parameter_count=parameters,
                    config=replace(config, train=replace(config.train, seed=seed)),
                )
    return entries, observations


def test_capacity_panel_computes_preregistered_contrasts(monkeypatch, tmp_path):
    entries, observations = _panel(tmp_path)
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_capacity_panel_observation",
        lambda path: observations[path],
    )

    analysis = analyze_capacity_panel(entries)

    assert set(analysis.cells) == {
        (corpus, arm)
        for corpus in ("frankenstein", "douglass", "origin")
        for arm in ("char128", "char136", "bytebpe512")
    }
    assert [
        value for _, value in analysis.contrasts["bytebpe512_minus_char136"]["frankenstein"]
    ] == pytest.approx([-0.06] * 3)
    assert analysis.parameter_gaps["frankenstein"] == pytest.approx(336 / 929_664)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda entries, observations: entries[:-1], "exactly 27"),
        (lambda entries, observations: [*entries[:-1], entries[0]], "distinct"),
        (
            lambda entries, observations: [
                (corpus, "wrong" if index == 0 else arm, path)
                for index, (corpus, arm, path) in enumerate(entries)
            ],
            "unknown",
        ),
    ],
)
def test_capacity_panel_rejects_invalid_design(monkeypatch, tmp_path, mutation, match):
    entries, observations = _panel(tmp_path)
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_capacity_panel_observation",
        lambda path: observations[path],
    )

    with pytest.raises(ValueError, match=match):
        analyze_capacity_panel(mutation(entries, observations))


def test_capacity_panel_rejects_parameter_mismatch(monkeypatch, tmp_path):
    entries, observations = _panel(tmp_path)
    target = next(
        path for corpus, arm, path in entries if arm == "char136" and path.name.endswith("1337")
    )
    observations[target] = replace(observations[target], parameter_count=960_000)
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_capacity_panel_observation",
        lambda path: observations[path],
    )

    with pytest.raises(ValueError, match="parameter gap"):
        analyze_capacity_panel(entries)


def test_load_capacity_panel_observation_combines_verified_artifacts(monkeypatch, tmp_path):
    entries, observations = _panel(tmp_path)
    _, _, path = entries[0]
    expected = observations[path]
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_run_summary",
        lambda run_dir: {"parameter_count": expected.parameter_count},
    )
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_config", lambda config_path: expected.config
    )
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_sealed_test_observation",
        lambda run_dir: expected.sealed,
    )

    loaded = load_capacity_panel_observation(path)

    assert loaded == expected

    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_sealed_test_observation",
        lambda run_dir: replace(expected.sealed, seed=2027),
    )
    with pytest.raises(ValueError, match="seed"):
        load_capacity_panel_observation(path)


@pytest.mark.parametrize("parameter_count", [None, True, 0, 1_000_000_001])
def test_load_capacity_panel_observation_rejects_parameter_count(
    monkeypatch, tmp_path, parameter_count
):
    entries, observations = _panel(tmp_path)
    _, _, path = entries[0]
    expected = observations[path]
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_run_summary",
        lambda run_dir: {"parameter_count": parameter_count},
    )
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_config", lambda config_path: expected.config
    )
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_sealed_test_observation",
        lambda run_dir: expected.sealed,
    )

    with pytest.raises(ValueError, match="parameter_count"):
        load_capacity_panel_observation(path)


def test_capacity_panel_rejects_arm_contract_and_unbalanced_seed(monkeypatch, tmp_path):
    entries, observations = _panel(tmp_path)
    target = entries[0][2]
    observations[target] = replace(
        observations[target],
        config=replace(
            observations[target].config,
            train=replace(observations[target].config.train, batch_size=99),
        ),
    )
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_capacity_panel_observation",
        lambda path: observations[path],
    )
    with pytest.raises(ValueError, match="tokenizer/width/context/batch"):
        analyze_capacity_panel(entries)

    entries, observations = _panel(tmp_path / "seeds")
    target = entries[0][2]
    observations[target] = replace(
        observations[target], sealed=replace(observations[target].sealed, seed=7)
    )
    monkeypatch.setattr(
        "smallm.evaluation.capacity_panel.load_capacity_panel_observation",
        lambda path: observations[path],
    )
    with pytest.raises(ValueError, match="three preregistered seeds"):
        analyze_capacity_panel(entries)


def test_capacity_panel_cli_supports_direct_execution():
    completed = subprocess.run(
        [sys.executable, "scripts/summarize_capacity_panel.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CORPUS ARM RUN_DIR" in completed.stdout
