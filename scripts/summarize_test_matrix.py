from __future__ import annotations

import argparse
from pathlib import Path

from smallm.evaluation.robustness import summarize_values
from smallm.evaluation.sealed_matrix import analyze_sealed_test_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell",
        action="append",
        nargs=3,
        metavar=("CORPUS", "TOKENIZER", "RUN_DIR"),
        required=True,
    )
    parser.add_argument("--reference-tokenizer", required=True)
    parser.add_argument("--candidate-tokenizer", required=True)
    args = parser.parse_args()
    entries = [(corpus, tokenizer, Path(path)) for corpus, tokenizer, path in args.cell]
    try:
        analysis = analyze_sealed_test_matrix(
            entries,
            reference_tokenizer=args.reference_tokenizer,
            candidate_tokenizer=args.candidate_tokenizer,
        )
    except (OSError, UnicodeError, TypeError, AttributeError, KeyError, ValueError) as exc:
        parser.error(str(exc))

    print("standard_deviation population_descriptive")
    print("contrast candidate_minus_reference_test_bpc")
    print()
    print("corpus tokenizer seed checkpoint_step test_bpc test_loss test_tokens target_characters")
    for (corpus, tokenizer), observations in sorted(analysis.cells.items()):
        for item in sorted(observations, key=lambda observation: observation.seed):
            print(
                f"{corpus} {tokenizer} {item.seed} {item.checkpoint_step} "
                f"{item.test_bpc:.6f} {item.test_loss:.6f} {item.test_tokens} "
                f"{item.target_characters}"
            )
    print()
    print("cell corpus tokenizer mean_bpc population_stddev min max")
    for (corpus, tokenizer), observations in sorted(analysis.cells.items()):
        summary = summarize_values([item.test_bpc for item in observations])
        print(
            f"cell {corpus} {tokenizer} {summary.mean:.6f} "
            f"{summary.population_stddev:.6f} {summary.minimum:.6f} {summary.maximum:.6f}"
        )
    print()
    print("paired corpus seed candidate_minus_reference_test_bpc")
    for corpus, contrasts in sorted(analysis.contrasts.items()):
        for seed, value in contrasts:
            print(f"paired {corpus} {seed} {value:.6f}")
        summary = summarize_values([value for _, value in contrasts])
        print(
            f"contrast_summary {corpus} {summary.mean:.6f} "
            f"{summary.population_stddev:.6f} {summary.minimum:.6f} {summary.maximum:.6f}"
        )
    print()
    for seed, value in analysis.interactions:
        print(f"interaction {seed} {value:.6f}")
    interaction = summarize_values([value for _, value in analysis.interactions])
    print(
        f"interaction_summary {interaction.mean:.6f} {interaction.population_stddev:.6f} "
        f"{interaction.minimum:.6f} {interaction.maximum:.6f}"
    )


if __name__ == "__main__":
    main()
