import pytest

from smallm.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig, load_config


def test_load_config_reads_weight_decay(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
train:
  learning_rate: 0.0001
  weight_decay: 0.01
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.train.learning_rate == 0.0001
    assert config.train.weight_decay == 0.01


def test_data_config_defaults_to_char_tokenizer():
    config = DataConfig()

    assert config.tokenizer_type == "char"
    assert config.bpe_vocab_size is None


def test_load_config_reads_bpe_tokenizer_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  tokenizer_type: bpe
  bpe_vocab_size: 128
  bpe_min_frequency: 3
""".strip()
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.data.tokenizer_type == "bpe"
    assert config.data.bpe_vocab_size == 128
    assert config.data.bpe_min_frequency == 3


def test_data_config_rejects_unknown_tokenizer_type():
    with pytest.raises(ValueError, match="tokenizer_type"):
        DataConfig(tokenizer_type="wordpiece")


def test_data_config_rejects_missing_bpe_vocab_size():
    with pytest.raises(ValueError, match="bpe_vocab_size"):
        DataConfig(tokenizer_type="bpe")


def test_data_config_rejects_invalid_bpe_vocab_size():
    with pytest.raises(ValueError, match="bpe_vocab_size"):
        DataConfig(tokenizer_type="bpe", bpe_vocab_size=0)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: DataConfig(block_size=0), "block_size"),
        (lambda: DataConfig(train_split=1.0), "train_split"),
        (
            lambda: DataConfig(tokenizer_type="bpe", bpe_vocab_size=10, bpe_min_frequency=0),
            "min_frequency",
        ),
        (lambda: ModelConfig(vocab_size=0), "vocab_size"),
        (lambda: ModelConfig(n_head=3, n_embd=8), "divisible"),
        (lambda: ModelConfig(dropout=1.0), "dropout"),
        (lambda: TrainConfig(run_name=" "), "run_name"),
        (lambda: TrainConfig(batch_size=0), "batch_size"),
        (lambda: TrainConfig(max_steps=0), "max_steps"),
        (lambda: TrainConfig(log_interval=0), "log_interval"),
        (lambda: TrainConfig(eval_interval=0), "eval_interval"),
        (lambda: TrainConfig(eval_batches=0), "eval_batches"),
        (lambda: TrainConfig(learning_rate=0), "learning_rate"),
        (lambda: TrainConfig(weight_decay=-1), "weight_decay"),
        (lambda: TrainConfig(sample_prompt=""), "sample_prompt"),
        (lambda: TrainConfig(sample_max_new_tokens=-1), "sample_max_new_tokens"),
        (lambda: TrainConfig(sample_temperature=0), "sample_temperature"),
        (lambda: TrainConfig(sample_top_k=0), "sample_top_k"),
        (
            lambda: ExperimentConfig(
                data=DataConfig(block_size=8), model=ModelConfig(block_size=4)
            ),
            "cannot exceed",
        ),
    ],
)
def test_config_rejects_invalid_values(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()
