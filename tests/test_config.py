import pytest

from smallm.config import DataConfig, load_config


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
