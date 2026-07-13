import pytest

from smallm.evaluation.robustness import RunObservation, summarize_observations, summarize_values


def test_summarize_values_reports_population_statistics():
    summary = summarize_values([1.0, 2.0, 3.0])

    assert summary.mean == 2.0
    assert summary.population_stddev == pytest.approx(0.81649658)
    assert summary.minimum == 1.0
    assert summary.maximum == 3.0


def test_summarize_values_rejects_empty_or_non_finite_input():
    with pytest.raises(ValueError, match="at least one"):
        summarize_values([])
    with pytest.raises(ValueError, match="finite"):
        summarize_values([float("nan")])


def test_summarize_observations_requires_distinct_seeds():
    observation = RunObservation(1, 10, 5, 2.0, 2.1, 3.0)

    with pytest.raises(ValueError, match="distinct seeds"):
        summarize_observations([observation, observation])


def test_summarize_observations_requires_multiple_comparable_runs():
    one = RunObservation(1, 10, 5, 2.0, 2.1, 3.0, "same")
    other = RunObservation(2, 10, 5, 2.0, 2.1, 3.0, "different")

    with pytest.raises(ValueError, match="at least two"):
        summarize_observations([one])
    with pytest.raises(ValueError, match="fingerprint"):
        summarize_observations([one, other])


def test_summarize_observations_covers_declared_metrics():
    observations = [
        RunObservation(1, 10, 5, 2.0, 2.1, 3.0),
        RunObservation(2, 20, 7, 2.2, 2.3, 5.0),
    ]

    summaries = summarize_observations(observations)

    assert set(summaries) == {
        "actual_steps",
        "best_step",
        "best_bpc",
        "final_bpc",
        "duration_seconds",
    }
    assert summaries["actual_steps"].mean == 15.0
    assert summaries["best_bpc"].mean == pytest.approx(2.1)
