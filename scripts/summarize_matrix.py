from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from smallm.evaluation.robustness import RunObservation, summarize_values
from smallm.evaluation.run_observation import load_run_artifact

_LABEL = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_MAX_CELLS = 64


@dataclass(frozen=True)
class MatrixAnalysis:
    cells: dict[tuple[str, str], list[RunObservation]]
    contrasts: dict[str, list[tuple[int, float]]]
    interactions: list[tuple[int, float]]


def analyze_matrix(
    entries: list[tuple[str, str, Path]],
    *,
    reference_tokenizer: str,
    candidate_tokenizer: str,
) -> MatrixAnalysis:
    if not 4 <= len(entries) <= _MAX_CELLS:
        raise ValueError(f"matrix requires between 4 and {_MAX_CELLS} run entries")
    if reference_tokenizer == candidate_tokenizer:
        raise ValueError("reference and candidate tokenizers must differ")
    if len({path.resolve() for _, _, path in entries}) != len(entries):
        raise ValueError("matrix run directories must be distinct")

    cells: dict[tuple[str, str], list[RunObservation]] = defaultdict(list)
    corpus_hashes: dict[str, set[str]] = defaultdict(set)
    for corpus, tokenizer, run_dir in entries:
        if not _LABEL.fullmatch(corpus) or not _LABEL.fullmatch(tokenizer):
            raise ValueError("corpus and tokenizer labels must be lowercase identifiers")
        summary, observation = load_run_artifact(run_dir)
        if summary.get("tokenizer_type") != tokenizer:
            raise ValueError(f"tokenizer label does not match run summary: {run_dir}")
        dataset = summary.get("dataset")
        prepared_sha256 = dataset.get("prepared_sha256") if isinstance(dataset, dict) else None
        if not isinstance(prepared_sha256, str) or len(prepared_sha256) != 64:
            raise ValueError(f"run summary has invalid prepared corpus checksum: {run_dir}")
        corpus_hashes[corpus].add(prepared_sha256)
        cells[(corpus, tokenizer)].append(observation)

    corpora = sorted(corpus_hashes)
    tokenizers = sorted({tokenizer for _, tokenizer in cells})
    if len(corpora) != 2 or set(tokenizers) != {reference_tokenizer, candidate_tokenizer}:
        raise ValueError("matrix must contain exactly two corpora and the declared two tokenizers")
    if any(len(hashes) != 1 for hashes in corpus_hashes.values()):
        raise ValueError("each corpus label must identify exactly one prepared corpus checksum")
    if len({next(iter(hashes)) for hashes in corpus_hashes.values()}) != len(corpus_hashes):
        raise ValueError(
            "different corpus labels must identify different prepared corpus checksums"
        )

    expected_keys = {(corpus, tokenizer) for corpus in corpora for tokenizer in tokenizers}
    if set(cells) != expected_keys:
        raise ValueError("matrix is missing a corpus-tokenizer cell")

    expected_seeds: set[int] | None = None
    for observations in cells.values():
        seeds = {observation.seed for observation in observations}
        if len(seeds) != len(observations) or len(seeds) < 2:
            raise ValueError("every matrix cell requires at least two distinct seeds")
        if len({observation.comparison_fingerprint for observation in observations}) != 1:
            raise ValueError("runs within a matrix cell do not share an experiment fingerprint")
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError("every matrix cell must contain the same seed set")

    contrasts: dict[str, list[tuple[int, float]]] = {}
    for corpus in corpora:
        reference = {item.seed: item.best_bpc for item in cells[(corpus, reference_tokenizer)]}
        candidate = {item.seed: item.best_bpc for item in cells[(corpus, candidate_tokenizer)]}
        contrasts[corpus] = [
            (seed, candidate[seed] - reference[seed]) for seed in sorted(reference)
        ]

    first, second = corpora
    first_deltas = dict(contrasts[first])
    second_deltas = dict(contrasts[second])
    interactions = [
        (seed, second_deltas[seed] - first_deltas[seed]) for seed in sorted(first_deltas)
    ]
    return MatrixAnalysis(dict(cells), contrasts, interactions)


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
        analysis = analyze_matrix(
            entries,
            reference_tokenizer=args.reference_tokenizer,
            candidate_tokenizer=args.candidate_tokenizer,
        )
    except (OSError, UnicodeError, TypeError, AttributeError, ValueError) as exc:
        parser.error(str(exc))

    print("standard_deviation population_descriptive")
    print("contrast candidate_minus_reference_bpc")
    print()
    print("corpus tokenizer seed actual_steps best_step best_bpc final_bpc duration_seconds")
    for (corpus, tokenizer), observations in sorted(analysis.cells.items()):
        for item in sorted(observations, key=lambda observation: observation.seed):
            print(
                f"{corpus} {tokenizer} {item.seed} {item.actual_steps} {item.best_step} "
                f"{item.best_bpc:.6f} {item.final_bpc:.6f} {item.duration_seconds:.1f}"
            )
    print()
    print("cell corpus tokenizer mean_bpc population_stddev min max")
    for (corpus, tokenizer), observations in sorted(analysis.cells.items()):
        summary = summarize_values([item.best_bpc for item in observations])
        print(
            f"cell {corpus} {tokenizer} {summary.mean:.6f} "
            f"{summary.population_stddev:.6f} {summary.minimum:.6f} {summary.maximum:.6f}"
        )
    print()
    print("paired corpus seed candidate_minus_reference_bpc")
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
