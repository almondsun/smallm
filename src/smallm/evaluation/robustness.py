from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import fmean, pstdev


@dataclass(frozen=True)
class RunObservation:
    seed: int
    actual_steps: int
    best_step: int
    best_bpc: float
    final_bpc: float
    duration_seconds: float
    comparison_fingerprint: str = "test-fixture"


@dataclass(frozen=True)
class DistributionSummary:
    mean: float
    population_stddev: float
    minimum: float
    maximum: float


def summarize_values(values: list[float]) -> DistributionSummary:
    if not values:
        raise ValueError("at least one value is required")
    if not all(isfinite(value) for value in values):
        raise ValueError("summary values must be finite")
    return DistributionSummary(
        mean=fmean(values),
        population_stddev=pstdev(values),
        minimum=min(values),
        maximum=max(values),
    )


def summarize_observations(
    observations: list[RunObservation],
) -> dict[str, DistributionSummary]:
    if len(observations) < 2:
        raise ValueError("at least two run observations are required")
    if len({observation.seed for observation in observations}) != len(observations):
        raise ValueError("run observations must have distinct seeds")
    if len({observation.comparison_fingerprint for observation in observations}) != 1:
        raise ValueError("run observations do not share an experiment fingerprint")
    return {
        "actual_steps": summarize_values(
            [float(observation.actual_steps) for observation in observations]
        ),
        "best_step": summarize_values(
            [float(observation.best_step) for observation in observations]
        ),
        "best_bpc": summarize_values([observation.best_bpc for observation in observations]),
        "final_bpc": summarize_values([observation.final_bpc for observation in observations]),
        "duration_seconds": summarize_values(
            [observation.duration_seconds for observation in observations]
        ),
    }
