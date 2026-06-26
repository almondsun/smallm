from smallm.config import load_config


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
