from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from smallm.evaluation.run_observation import load_run_artifact

_LABEL = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_ARTIFACT_BYTES = 1_000_000
_MAX_ENTRIES = 64


@dataclass(frozen=True)
class SealedTestObservation:
    seed: int
    tokenizer: str
    prepared_sha256: str
    comparison_fingerprint: str
    checkpoint_step: int
    checkpoint_sha256: str
    test_loss: float
    test_bpc: float
    test_tokens: int
    target_tokens: int
    target_characters: int


@dataclass(frozen=True)
class SealedMatrixAnalysis:
    cells: dict[tuple[str, str], list[SealedTestObservation]]
    contrasts: dict[str, list[tuple[int, float]]]
    interactions: list[tuple[int, float]]


def _read_json_object(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(_MAX_ARTIFACT_BYTES + 1)
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError(f"sealed test artifact exceeds {_MAX_ARTIFACT_BYTES} bytes: {path}")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"sealed test artifact must contain a JSON object: {path}")
    return payload


def _bounded_int(payload: dict[str, Any], field: str, *, positive: bool = False) -> int:
    value = payload.get(field)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < int(positive)
        or value > 1_000_000_000
    ):
        raise ValueError(f"sealed test field {field!r} must be a bounded integer")
    return value


def _bounded_float(payload: dict[str, Any], field: str, *, positive: bool = False) -> float:
    value = payload.get(field)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"sealed test field {field!r} must be numeric")
    number = float(value)
    if not isfinite(number) or number < 0 or (positive and number == 0) or number > 1e9:
        raise ValueError(f"sealed test field {field!r} is outside supported bounds")
    return number


def load_sealed_test_observation(run_dir: Path) -> SealedTestObservation:
    summary, validation = load_run_artifact(run_dir)
    path = run_dir / "test_evaluation_best.json"
    payload = _read_json_object(path)
    if payload.get("schema_version") != 1 or payload.get("status") != "complete":
        raise ValueError(f"sealed test artifact is not a complete schema-v1 result: {path}")
    if payload.get("checkpoint_kind") != "best":
        raise ValueError("sealed matrix requires best-checkpoint test evaluations")
    if payload.get("evaluation_mode") != "full" or _bounded_float(payload, "test_coverage") != 1.0:
        raise ValueError("sealed matrix requires full test coverage")

    dataset = summary.get("dataset")
    prepared_sha256 = dataset.get("prepared_sha256") if isinstance(dataset, dict) else None
    if not isinstance(prepared_sha256, str) or not _SHA256.fullmatch(prepared_sha256):
        raise ValueError("run summary has an invalid prepared corpus checksum")
    if payload.get("prepared_sha256") != prepared_sha256:
        raise ValueError("sealed test corpus checksum does not match the run summary")
    checkpoint_step = _bounded_int(payload, "checkpoint_step")
    if checkpoint_step != validation.best_step:
        raise ValueError("sealed test checkpoint step does not match best validation step")
    test_characters = _bounded_int(payload, "test_characters", positive=True)
    if (
        summary.get("test_status") != "sealed_unread"
        or summary.get("test_characters") != test_characters
    ):
        raise ValueError("sealed test support does not match the run summary")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha256, str) or not _SHA256.fullmatch(checkpoint_sha256):
        raise ValueError("sealed test checkpoint checksum must be lowercase SHA-256")
    test_tokens = _bounded_int(payload, "test_tokens", positive=True)
    target_tokens = _bounded_int(payload, "test_target_tokens", positive=True)
    total_target_tokens = _bounded_int(payload, "test_total_target_tokens", positive=True)
    if target_tokens != total_target_tokens or target_tokens >= test_tokens:
        raise ValueError("sealed test target-token support is inconsistent")
    target_characters = _bounded_int(payload, "test_target_characters", positive=True)
    if target_characters > test_characters:
        raise ValueError("sealed test target-character support is inconsistent")
    tokenizer = summary.get("tokenizer_type")
    if not isinstance(tokenizer, str):
        raise ValueError("run summary tokenizer type must be a string")
    return SealedTestObservation(
        seed=validation.seed,
        tokenizer=tokenizer,
        prepared_sha256=prepared_sha256,
        comparison_fingerprint=validation.comparison_fingerprint,
        checkpoint_step=checkpoint_step,
        checkpoint_sha256=checkpoint_sha256,
        test_loss=_bounded_float(payload, "test_loss", positive=True),
        test_bpc=_bounded_float(payload, "test_bits_per_character", positive=True),
        test_tokens=test_tokens,
        target_tokens=target_tokens,
        target_characters=target_characters,
    )


def analyze_sealed_test_matrix(
    entries: list[tuple[str, str, Path]],
    *,
    reference_tokenizer: str,
    candidate_tokenizer: str,
) -> SealedMatrixAnalysis:
    if not 4 <= len(entries) <= _MAX_ENTRIES:
        raise ValueError(f"matrix requires between 4 and {_MAX_ENTRIES} run entries")
    if reference_tokenizer == candidate_tokenizer:
        raise ValueError("reference and candidate tokenizers must differ")
    if len({path.resolve() for _, _, path in entries}) != len(entries):
        raise ValueError("matrix run directories must be distinct")

    cells: dict[tuple[str, str], list[SealedTestObservation]] = defaultdict(list)
    corpus_hashes: dict[str, set[str]] = defaultdict(set)
    for corpus, tokenizer, run_dir in entries:
        if not _LABEL.fullmatch(corpus) or not _LABEL.fullmatch(tokenizer):
            raise ValueError("corpus and tokenizer labels must be lowercase identifiers")
        observation = load_sealed_test_observation(run_dir)
        if observation.tokenizer != tokenizer:
            raise ValueError(f"tokenizer label does not match run summary: {run_dir}")
        cells[(corpus, tokenizer)].append(observation)
        corpus_hashes[corpus].add(observation.prepared_sha256)

    corpora = sorted(corpus_hashes)
    tokenizers = {tokenizer for _, tokenizer in cells}
    if len(corpora) != 2 or tokenizers != {reference_tokenizer, candidate_tokenizer}:
        raise ValueError("matrix must contain exactly two corpora and the declared two tokenizers")
    if any(len(hashes) != 1 for hashes in corpus_hashes.values()):
        raise ValueError("each corpus label must identify exactly one prepared corpus checksum")
    if len({next(iter(hashes)) for hashes in corpus_hashes.values()}) != len(corpora):
        raise ValueError("different corpus labels must identify different prepared checksums")
    expected_cells = {(corpus, tokenizer) for corpus in corpora for tokenizer in sorted(tokenizers)}
    if set(cells) != expected_cells:
        raise ValueError("matrix is missing a corpus-tokenizer cell")

    expected_seeds: set[int] | None = None
    for observations in cells.values():
        seeds = {item.seed for item in observations}
        if len(seeds) != len(observations) or len(seeds) < 2:
            raise ValueError("every matrix cell requires at least two distinct seeds")
        if len({item.comparison_fingerprint for item in observations}) != 1:
            raise ValueError("runs within a matrix cell do not share an experiment fingerprint")
        if expected_seeds is None:
            expected_seeds = seeds
        elif seeds != expected_seeds:
            raise ValueError("every matrix cell must contain the same seed set")

    contrasts: dict[str, list[tuple[int, float]]] = {}
    for corpus in corpora:
        reference = {item.seed: item.test_bpc for item in cells[(corpus, reference_tokenizer)]}
        candidate = {item.seed: item.test_bpc for item in cells[(corpus, candidate_tokenizer)]}
        contrasts[corpus] = [
            (seed, candidate[seed] - reference[seed]) for seed in sorted(reference)
        ]
    first, second = corpora
    first_deltas = dict(contrasts[first])
    second_deltas = dict(contrasts[second])
    interactions = [
        (seed, second_deltas[seed] - first_deltas[seed]) for seed in sorted(first_deltas)
    ]
    return SealedMatrixAnalysis(dict(cells), contrasts, interactions)
