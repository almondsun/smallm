from __future__ import annotations

import argparse
from pathlib import Path

from smallm.evaluation.robustness import summarize_observations
from smallm.evaluation.run_observation import load_run_observation, load_run_summary

_MAX_RUNS = 32

_load_summary = load_run_summary
_observation = load_run_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    if not 2 <= len(args.runs) <= _MAX_RUNS:
        parser.error(f"provide between 2 and {_MAX_RUNS} run directories")
    observations = [_observation(run_dir) for run_dir in args.runs]
    summaries = summarize_observations(observations)

    print(f"verified_experiment_fingerprint_sha256 {observations[0].comparison_fingerprint}")
    print("standard_deviation population_descriptive")
    print()
    print("seed actual_steps best_step best_bpc final_bpc duration_seconds")
    for observation in observations:
        print(
            f"{observation.seed} {observation.actual_steps} {observation.best_step} "
            f"{observation.best_bpc:.6f} {observation.final_bpc:.6f} "
            f"{observation.duration_seconds:.1f}"
        )
    print()
    print("metric mean population_stddev min max")
    for metric, summary in summaries.items():
        print(
            f"{metric} {summary.mean:.6f} {summary.population_stddev:.6f} "
            f"{summary.minimum:.6f} {summary.maximum:.6f}"
        )


if __name__ == "__main__":
    main()
