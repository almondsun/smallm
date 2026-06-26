from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingRunInfo:
    project: str
    device: str
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    batch_size: int
    max_steps: int
    parameter_count: int
    checkpoint_path: str


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def format_loss(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def format_progress_row(
    *,
    step: int,
    max_steps: int,
    train_loss: float,
    val_loss: float | None,
    learning_rate: float,
    elapsed_seconds: float,
    tokens_per_second: float,
) -> str:
    steps_per_second = step / elapsed_seconds if elapsed_seconds > 0 else 0.0
    remaining_steps = max_steps - step
    eta_seconds = remaining_steps / steps_per_second if steps_per_second > 0 else 0.0
    return (
        f"{step:>6d}/{max_steps:<6d} "
        f"{train_loss:>10.4f} "
        f"{format_loss(val_loss):>10s} "
        f"{learning_rate:>10.2e} "
        f"{format_duration(elapsed_seconds):>8s} "
        f"{tokens_per_second:>11.0f} "
        f"{format_duration(eta_seconds):>8s}"
    )


class TrainingProgressLogger:
    def __init__(self) -> None:
        self._width = 84

    def header(self, info: TrainingRunInfo) -> None:
        print("smaLLM training progress logger")
        print("=" * self._width)
        print(f"{'project':>16}: {info.project}")
        print(f"{'device':>16}: {info.device}")
        print(f"{'vocab size':>16}: {info.vocab_size}")
        print(f"{'block size':>16}: {info.block_size}")
        print(f"{'layers':>16}: {info.n_layer}")
        print(f"{'heads':>16}: {info.n_head}")
        print(f"{'embedding dim':>16}: {info.n_embd}")
        print(f"{'batch size':>16}: {info.batch_size}")
        print(f"{'max steps':>16}: {info.max_steps}")
        print(f"{'parameters':>16}: {info.parameter_count:,}")
        print(f"{'checkpoint':>16}: {info.checkpoint_path}")
        print("-" * self._width)
        print(
            f"{'step':>13s} "
            f"{'train_loss':>10s} "
            f"{'val_loss':>10s} "
            f"{'lr':>10s} "
            f"{'elapsed':>8s} "
            f"{'tokens/sec':>11s} "
            f"{'eta':>8s}"
        )
        print("-" * self._width, flush=True)

    def progress(
        self,
        *,
        step: int,
        max_steps: int,
        train_loss: float,
        val_loss: float | None,
        learning_rate: float,
        elapsed_seconds: float,
        tokens_per_second: float,
    ) -> None:
        print(
            format_progress_row(
                step=step,
                max_steps=max_steps,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=learning_rate,
                elapsed_seconds=elapsed_seconds,
                tokens_per_second=tokens_per_second,
            ),
            flush=True,
        )

    def summary(self, *, checkpoint_path: str, elapsed_seconds: float, final_loss: float) -> None:
        print("-" * self._width)
        print(f"training complete in {format_duration(elapsed_seconds)}")
        print(f"final train loss: {final_loss:.4f}")
        print(f"checkpoint saved: {checkpoint_path}", flush=True)
