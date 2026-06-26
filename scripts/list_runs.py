from __future__ import annotations

import argparse
from pathlib import Path

from smallm.training.logging import format_duration, format_loss
from smallm.training.runs import list_all_run_dirs, list_run_dirs, load_last_metric, load_summary


def _short(path: Path) -> str:
    return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--run-name")
    args = parser.parse_args()

    if args.run_name:
        run_dirs = list_run_dirs(args.run_name, args.runs_dir)
    else:
        run_dirs = list_all_run_dirs(args.runs_dir)

    print(
        f"{'run_name':<12} {'run_id':<20} {'train':>10} {'val':>10} "
        f"{'duration':>9} {'params':>10} checkpoint"
    )
    print("-" * 100)
    for run_dir in run_dirs:
        summary = load_summary(run_dir)
        metric = load_last_metric(run_dir) or {}
        train_loss = summary.get("final_train_loss", metric.get("train_loss"))
        val_loss = summary.get("final_val_loss", metric.get("val_loss"))
        checkpoint_path = summary.get("checkpoint_path", str(run_dir / "checkpoint.pt"))
        print(
            f"{run_dir.parent.name:<12} {run_dir.name:<20} "
            f"{format_loss(train_loss):>10} {format_loss(val_loss):>10} "
            f"{format_duration(summary.get('duration_seconds', 0.0)):>9} "
            f"{summary.get('parameter_count', 0):>10,} {_short(Path(checkpoint_path))}"
        )


if __name__ == "__main__":
    main()
