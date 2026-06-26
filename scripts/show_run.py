from __future__ import annotations

import argparse
from pathlib import Path

from smallm.training.runs import load_last_metric, load_summary, resolve_run_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    run_dir = resolve_run_path(args.run, run_name=args.run_name, runs_dir=args.runs_dir)
    summary = load_summary(run_dir)
    last_metric = load_last_metric(run_dir)
    sample_path = Path(summary.get("sample_path", run_dir / "sample.txt"))
    sample_text = sample_path.read_text(encoding="utf-8") if sample_path.exists() else ""

    print(f"run: {run_dir}")
    print(f"config: {summary.get('config_path', run_dir / 'config.yaml')}")
    print(f"metrics: {summary.get('metrics_path', run_dir / 'metrics.jsonl')}")
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"checkpoint: {summary.get('checkpoint_path', run_dir / 'checkpoint.pt')}")
    dataset = summary.get("dataset")
    if dataset:
        prepared_sha = dataset.get("prepared_sha256") or ""
        print("dataset:")
        print(f"  source: {dataset.get('source_name')}")
        print(f"  prepared_sha256: {prepared_sha[:12]}")
        print(f"  prepared_characters: {dataset.get('prepared_characters')}")
        print(f"  unique_characters: {dataset.get('unique_characters')}")
        print(
            "  split_characters: "
            f"train={dataset.get('train_characters')} "
            f"val={dataset.get('validation_characters')}"
        )
    if last_metric is not None:
        print(
            "last metric: "
            f"step={last_metric.get('step')} "
            f"train_loss={last_metric.get('train_loss')} "
            f"val_loss={last_metric.get('val_loss')}"
        )
    if summary.get("best_val_loss") is not None:
        print(
            "best validation: "
            f"step={summary.get('best_val_step')} "
            f"val_loss={summary.get('best_val_loss')}"
        )
    print("sample:")
    print(sample_text)


if __name__ == "__main__":
    main()
