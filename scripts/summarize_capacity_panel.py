from __future__ import annotations

import argparse
from pathlib import Path

from smallm.evaluation.capacity_panel import analyze_capacity_panel
from smallm.evaluation.robustness import summarize_values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        action="append",
        nargs=3,
        metavar=("CORPUS", "ARM", "RUN_DIR"),
        required=True,
    )
    args = parser.parse_args()
    entries = [(corpus, arm, Path(path)) for corpus, arm, path in args.cell]
    try:
        analysis = analyze_capacity_panel(entries)
    except (OSError, UnicodeError, TypeError, AttributeError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    print("standard_deviation population_descriptive")
    print("contrast candidate_minus_reference_test_bpc")
    print()
    print("cell corpus arm seed parameters test_bpc checkpoint_step")
    for (corpus, arm), observations in sorted(analysis.cells.items()):
        for item in sorted(observations, key=lambda value: value.sealed.seed):
            print(
                f"cell {corpus} {arm} {item.sealed.seed} {item.parameter_count} "
                f"{item.sealed.test_bpc:.6f} {item.sealed.checkpoint_step}"
            )
    print()
    for corpus, gap in sorted(analysis.parameter_gaps.items()):
        print(f"parameter_gap {corpus} char136_minus_bytebpe512_fraction {gap:.6f}")
    print()
    for name, by_corpus in analysis.contrasts.items():
        for corpus, values in sorted(by_corpus.items()):
            for seed, value in values:
                print(f"paired {name} {corpus} {seed} {value:.6f}")
            summary = summarize_values([value for _, value in values])
            print(
                f"contrast_summary {name} {corpus} {summary.mean:.6f} "
                f"{summary.population_stddev:.6f} {summary.minimum:.6f} {summary.maximum:.6f}"
            )


if __name__ == "__main__":
    main()
