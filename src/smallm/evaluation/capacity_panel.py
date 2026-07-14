from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from smallm.config import ExperimentConfig, load_config
from smallm.evaluation.run_observation import load_run_summary
from smallm.evaluation.sealed_matrix import SealedTestObservation, load_sealed_test_observation

ARMS = ("char128", "char136", "bytebpe512")
SEEDS = (1337, 2027, 4242)
PARAMETER_MATCH_TOLERANCE = 0.015


@dataclass(frozen=True)
class CapacityPanelObservation:
    sealed: SealedTestObservation
    parameter_count: int
    config: ExperimentConfig


@dataclass(frozen=True)
class CapacityPanelAnalysis:
    cells: dict[tuple[str, str], list[CapacityPanelObservation]]
    contrasts: dict[str, dict[str, list[tuple[int, float]]]]
    parameter_gaps: dict[str, float]


def _bounded_parameter_count(summary: dict[str, object]) -> int:
    value = summary.get("parameter_count")
    if not isinstance(value, int) or isinstance(value, bool) or not 0 < value <= 1_000_000_000:
        raise ValueError("run summary parameter_count must be a bounded positive integer")
    return value


def load_capacity_panel_observation(run_dir: Path) -> CapacityPanelObservation:
    summary = load_run_summary(run_dir)
    try:
        config = load_config(run_dir / "config.yaml")
    except (OSError, UnicodeError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError(
            f"invalid capacity-panel config at {run_dir / 'config.yaml'}: {exc}"
        ) from exc
    sealed = load_sealed_test_observation(run_dir)
    if sealed.seed != config.train.seed:
        raise ValueError("sealed observation seed does not match run config")
    return CapacityPanelObservation(
        sealed=sealed,
        parameter_count=_bounded_parameter_count(summary),
        config=config,
    )


def _validate_arm(arm: str, observation: CapacityPanelObservation) -> None:
    config = observation.config
    expected = {
        "char128": ("char", None, 64, 128, 16),
        "char136": ("char", None, 64, 136, 16),
        "bytebpe512": ("byte_bpe", 512, 37, 128, 27),
    }[arm]
    actual = (
        config.data.tokenizer_type,
        config.data.bpe_vocab_size,
        config.data.block_size,
        config.model.n_embd,
        config.train.batch_size,
    )
    if actual != expected:
        raise ValueError(
            f"arm {arm} does not match its preregistered tokenizer/width/context/batch"
        )
    if observation.sealed.tokenizer != config.data.tokenizer_type:
        raise ValueError(f"arm {arm} tokenizer does not match the sealed artifact")
    if (
        config.data.train_split != 0.8
        or config.data.validation_split != 0.1
        or config.model.block_size != config.data.block_size
        or config.model.n_layer != 4
        or config.model.n_head != 4
        or config.model.dropout != 0.1
        or config.train.max_steps != 5000
        or config.train.learning_rate != 0.001
        or config.train.weight_decay != 0.0
        or config.train.eval_interval != 250
        or config.train.eval_batches is not None
        or config.train.early_stopping_patience != 3
        or config.train.early_stopping_min_delta != 0.0
    ):
        raise ValueError(f"arm {arm} does not match the common preregistered training contract")


def analyze_capacity_panel(
    entries: list[tuple[str, str, Path]],
) -> CapacityPanelAnalysis:
    if len(entries) != 27:
        raise ValueError("capacity panel requires exactly 27 run entries")
    if len({path.resolve() for _, _, path in entries}) != len(entries):
        raise ValueError("capacity panel run directories must be distinct")

    cells: dict[tuple[str, str], list[CapacityPanelObservation]] = {}
    corpus_hashes: dict[str, set[str]] = {}
    for corpus, arm, run_dir in entries:
        if arm not in ARMS:
            raise ValueError(f"unknown capacity-panel arm: {arm}")
        if not corpus or not corpus.replace("_", "").isalnum() or not corpus[0].islower():
            raise ValueError(f"invalid corpus label: {corpus}")
        observation = load_capacity_panel_observation(run_dir)
        _validate_arm(arm, observation)
        cells.setdefault((corpus, arm), []).append(observation)
        corpus_hashes.setdefault(corpus, set()).add(observation.sealed.prepared_sha256)

    corpora = sorted(corpus_hashes)
    if len(corpora) != 3:
        raise ValueError("capacity panel requires exactly three corpora")
    expected_cells = {(corpus, arm) for corpus in corpora for arm in ARMS}
    if set(cells) != expected_cells:
        raise ValueError("capacity panel is missing a corpus-arm cell")
    if any(len(hashes) != 1 for hashes in corpus_hashes.values()):
        raise ValueError("each corpus label must identify exactly one prepared corpus checksum")
    if len({next(iter(hashes)) for hashes in corpus_hashes.values()}) != 3:
        raise ValueError("different corpus labels must identify different prepared checksums")

    for key, observations in cells.items():
        seeds = {item.sealed.seed for item in observations}
        if seeds != set(SEEDS) or len(observations) != len(SEEDS):
            raise ValueError(f"cell {key} must contain the three preregistered seeds")
        if len({item.sealed.comparison_fingerprint for item in observations}) != 1:
            raise ValueError(f"cell {key} does not share one experiment fingerprint")

    contrasts: dict[str, dict[str, list[tuple[int, float]]]] = {
        "bytebpe512_minus_char136": {},
        "bytebpe512_minus_char128": {},
        "char136_minus_char128": {},
    }
    parameter_gaps: dict[str, float] = {}
    for corpus in corpora:
        by_arm = {arm: {item.sealed.seed: item for item in cells[(corpus, arm)]} for arm in ARMS}
        for seed in SEEDS:
            matched = by_arm["char136"][seed].parameter_count
            candidate = by_arm["bytebpe512"][seed].parameter_count
            gap = (matched - candidate) / candidate
            if abs(gap) > PARAMETER_MATCH_TOLERANCE:
                raise ValueError(
                    f"{corpus} seed {seed} parameter gap {gap:.4%} exceeds "
                    f"{PARAMETER_MATCH_TOLERANCE:.1%}"
                )
        parameter_gaps[corpus] = (
            by_arm["char136"][SEEDS[0]].parameter_count
            - by_arm["bytebpe512"][SEEDS[0]].parameter_count
        ) / by_arm["bytebpe512"][SEEDS[0]].parameter_count
        contrasts["bytebpe512_minus_char136"][corpus] = [
            (
                seed,
                by_arm["bytebpe512"][seed].sealed.test_bpc
                - by_arm["char136"][seed].sealed.test_bpc,
            )
            for seed in SEEDS
        ]
        contrasts["bytebpe512_minus_char128"][corpus] = [
            (
                seed,
                by_arm["bytebpe512"][seed].sealed.test_bpc
                - by_arm["char128"][seed].sealed.test_bpc,
            )
            for seed in SEEDS
        ]
        contrasts["char136_minus_char128"][corpus] = [
            (
                seed,
                by_arm["char136"][seed].sealed.test_bpc - by_arm["char128"][seed].sealed.test_bpc,
            )
            for seed in SEEDS
        ]
    return CapacityPanelAnalysis(cells=cells, contrasts=contrasts, parameter_gaps=parameter_gaps)
