from __future__ import annotations

import argparse

from smallm.evaluation import evaluate_sealed_test
from smallm.training.runs import resolve_run_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--checkpoint-kind", choices=("best", "final"), default="best")
    args = parser.parse_args()
    try:
        run_dir = resolve_run_path(args.run, run_name=args.run_name, runs_dir=args.runs_dir)
        output_path, payload = evaluate_sealed_test(run_dir, checkpoint_kind=args.checkpoint_kind)
    except (OSError, UnicodeError, TypeError, AttributeError, KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(f"sealed test evaluation: {output_path}")
    print(f"test loss: {payload['test_loss']:.6f}")
    print(f"test BPC: {payload['test_bits_per_character']:.6f}")
    print(f"coverage: {payload['test_coverage']:.6f}")


if __name__ == "__main__":
    main()
