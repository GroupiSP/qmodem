"""Seam 3: the evaluation driver's pure core (predictive and reference samples per
test case -> metric table and aggregates)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from stagereg.core import expected_sample_crps, wsu

from qmodem.battery.evaluate import CaseSamples, aggregate, case_metrics, metric_table

_FINITE = st.floats(-50.0, 50.0, allow_nan=False, width=32)


@st.composite
def case_samples(draw: st.DrawFn, *, name: str = "case_0", policy: int = 0):
    """A test case with t0 plus at least two evaluated timestamps."""
    n_times = draw(st.integers(3, 7))
    n_pred = draw(st.integers(2, 12))
    n_ref = draw(st.integers(2, 12))
    steps = draw(st.lists(st.floats(0.5, 20.0), min_size=n_times, max_size=n_times))
    arrays = st.lists(_FINITE, min_size=1, max_size=1).map(lambda v: v[0])

    def matrix(rows: int, cols: int) -> np.ndarray:
        values = draw(st.lists(arrays, min_size=rows * cols, max_size=rows * cols))
        return np.asarray(values, dtype=np.float64).reshape(rows, cols)

    predictive = matrix(n_times, n_pred)
    predictive[0] = np.nan  # t0 has no prediction
    return CaseSamples(
        name=name,
        policy=policy,
        times=np.cumsum(steps),
        targets=matrix(n_times, 1)[:, 0],
        reference_samples=matrix(n_times, n_ref),
        predictive_samples=predictive,
    )


@settings(max_examples=50, deadline=None)
@given(case=case_samples(), t0_values=st.data())
def test_the_t0_row_never_changes_a_metric(case: CaseSamples, t0_values) -> None:
    other = t0_values.draw(case_samples())
    n_pred = case.predictive_samples.shape[1]
    n_ref = case.reference_samples.shape[1]
    replaced = dataclasses.replace(
        case,
        targets=np.concatenate([other.targets[:1], case.targets[1:]]),
        reference_samples=np.vstack(
            [np.resize(other.reference_samples[0], n_ref), case.reference_samples[1:]]
        ),
        predictive_samples=np.vstack(
            [np.full((1, n_pred), 1e6), case.predictive_samples[1:]]
        ),
    )

    assert case_metrics(replaced, coverage_levels=(0.5, 0.95)) == case_metrics(
        case, coverage_levels=(0.5, 0.95)
    )


@settings(max_examples=50, deadline=None)
@given(case=case_samples(), seed=st.integers(0, 2**32 - 1))
def test_metrics_are_invariant_to_permuting_samples_within_a_timestamp(
    case: CaseSamples, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    permuted = dataclasses.replace(
        case,
        predictive_samples=rng.permuted(case.predictive_samples, axis=1),
    )

    expected = case_metrics(case, coverage_levels=(0.95,))
    actual = case_metrics(permuted, coverage_levels=(0.95,))
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=1e-9, abs=1e-9)


@settings(max_examples=50, deadline=None)
@given(
    case=case_samples(),
    levels=st.lists(st.floats(0.01, 0.99), min_size=2, max_size=6, unique=True),
)
def test_coverage_is_monotone_non_decreasing_in_the_level(
    case: CaseSamples, levels: list[float]
) -> None:
    levels = sorted(levels)
    metrics = case_metrics(case, coverage_levels=levels)

    coverages = [metrics[f"coverage_{100 * level:g}"] for level in levels]
    assert coverages == sorted(coverages)


@settings(max_examples=30, deadline=None)
@given(case=case_samples())
def test_crps_is_the_mean_expected_sample_crps_without_t0(case: CaseSamples) -> None:
    expected = expected_sample_crps(
        case.predictive_samples[1:], case.reference_samples[1:]
    ).mean()

    assert case_metrics(case, coverage_levels=(0.95,))["crps"] == pytest.approx(
        expected
    )


@settings(max_examples=30, deadline=None)
@given(case=case_samples())
def test_wsu_is_stageregs_per_trajectory_wsu_with_t0_as_origin(
    case: CaseSamples,
) -> None:
    lower, upper = np.percentile(case.predictive_samples[1:], [2.5, 97.5], axis=1)
    expected = wsu(
        np.concatenate([lower[:1], lower]),
        np.concatenate([upper[:1], upper]),
        case.times,
    )

    assert case_metrics(case, coverage_levels=(0.95,))["wsu"] == pytest.approx(expected)


def test_rmse_and_coverage_on_a_worked_example() -> None:
    case = CaseSamples(
        name="case_0",
        policy=0,
        times=np.array([0.0, 10.0, 20.0, 30.0]),
        targets=np.array([99.0, 1.0, 5.0, 100.0]),
        reference_samples=np.zeros((4, 3)),
        predictive_samples=np.array(
            [[np.nan] * 3, [0.0, 1.0, 2.0], [2.0, 3.0, 4.0], [4.0, 5.0, 6.0]]
        ),
    )

    metrics = case_metrics(case, coverage_levels=(0.95,))

    # sample means 1, 3, 5 vs targets 1, 5, 100: squared errors 0, 4, 9025
    assert metrics["rmse"] == pytest.approx(np.sqrt((0 + 4 + 9025) / 3))
    # only the first evaluated target lies inside its 95% band
    assert metrics["coverage_95"] == pytest.approx(1 / 3)


_METRIC_VALUES = st.fixed_dictionaries(
    {name: st.floats(0.0, 1e3) for name in ("crps", "rmse", "coverage_95", "wsu")}
)


@given(
    rows=st.lists(st.tuples(st.integers(0, 2), _METRIC_VALUES), min_size=1, max_size=8)
)
def test_the_aggregate_is_the_unweighted_mean_and_policies_partition_the_cases(
    rows: list[tuple[int, dict[str, float]]],
) -> None:
    table = {f"case_{i}": metrics for i, (_, metrics) in enumerate(rows)}
    policies = {f"case_{i}": policy for i, (policy, _) in enumerate(rows)}

    result = aggregate(table, policies)

    counts = {p: list(policies.values()).count(p) for p in set(policies.values())}
    assert set(result.per_policy) == set(counts)
    for metric in ("crps", "rmse", "coverage_95", "wsu"):
        values = [metrics[metric] for _, metrics in rows]
        assert result.overall[metric] == pytest.approx(sum(values) / len(values))
        recombined = sum(
            counts[p] * result.per_policy[p][metric] for p in counts
        ) / len(rows)
        assert recombined == pytest.approx(result.overall[metric])


def test_the_metric_table_has_one_row_per_case() -> None:
    cases = [
        CaseSamples(
            name=f"case_{i}",
            policy=i % 2,
            times=np.array([0.0, 1.0, 2.0]),
            targets=np.array([0.0, 1.0, 2.0]),
            reference_samples=np.ones((3, 2)),
            predictive_samples=np.ones((3, 4)) * i,
        )
        for i in range(3)
    ]

    table = metric_table(cases, coverage_levels=(0.5, 0.95))

    assert list(table) == ["case_0", "case_1", "case_2"]
    assert table["case_1"] == case_metrics(cases[1], coverage_levels=(0.5, 0.95))
