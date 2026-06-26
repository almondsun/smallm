from smallm.training.logging import format_duration, format_progress_row


def test_format_duration_uses_compact_units():
    assert format_duration(4.9) == "4s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3661) == "1h01m01s"


def test_format_progress_row_contains_aligned_metrics():
    row = format_progress_row(
        step=10,
        max_steps=100,
        train_loss=2.34567,
        val_loss=None,
        learning_rate=3e-4,
        elapsed_seconds=5.0,
        tokens_per_second=1024.0,
    )

    assert "10/100" in row
    assert "2.3457" in row
    assert "3.00e-04" in row
    assert "1024" in row
    assert "45s" in row
