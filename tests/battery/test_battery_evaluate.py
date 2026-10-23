"""Seam 3: the evaluation driver (predictive and reference samples per test case ->
metric table and aggregates, plus the sampling and logging around that core)."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from matplotlib.figure import Figure
from stagereg.core import expected_sample_crps, wsu
from stagereg.jax import PosteriorPredictive

from qmodem.battery.evaluate import (
    CaseSamples,
    EvaluationParameters,
    aggregate,
    case_metrics,
    evaluate_predictive_model,
    evaluation_indices,
    log_evaluation,
    metric_table,
)
from qmodem.battery.forecasts import ReferenceSamples

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
        reference=ReferenceSamples(matrix(n_times, n_ref)),
        predictive_samples=predictive,
    )


@settings(max_examples=50, deadline=None)
@given(case=case_samples(), t0_values=st.data())
def test_the_t0_row_never_changes_a_metric(case: CaseSamples, t0_values) -> None:
    other = t0_values.draw(case_samples())
    n_pred = case.predictive_samples.shape[1]
    n_ref = case.reference.samples.shape[1]
    replaced = dataclasses.replace(
        case,
        targets=np.concatenate([other.targets[:1], case.targets[1:]]),
        reference=ReferenceSamples(
            np.vstack(
                [
                    np.resize(other.reference.samples[0], n_ref),
                    case.reference.samples[1:],
                ]
            )
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
        case.predictive_samples[1:], case.reference.samples[1:]
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
        reference=ReferenceSamples(np.zeros((4, 3))),
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
            reference=ReferenceSamples(np.ones((3, 2))),
            predictive_samples=np.ones((3, 4)) * i,
        )
        for i in range(3)
    ]

    table = metric_table(cases, coverage_levels=(0.5, 0.95))

    assert list(table) == ["case_0", "case_1", "case_2"]
    assert table["case_1"] == case_metrics(cases[1], coverage_levels=(0.5, 0.95))


# --- Driver: a predictive model + test histories -> samples, table, logging ------

N_TIMESTAMPS = 6
N_REFERENCE = 5
OFFSETS = np.linspace(-1.0, 1.0, 8)
ALEATORIC_VARIANCE = 0.25


class _OffsetModel:
    """Draw s predicts the window's last load plus ``OFFSETS[s % 8]``."""

    def predictive(self, inputs, key, num_draws):
        last_load = inputs[:, -1, :1]
        offsets = jnp.asarray(np.resize(OFFSETS, num_draws))[:, None, None]
        return PosteriorPredictive(
            draws=last_load[None] + offsets,
            aleatoric_variance=jnp.full(last_load.shape, ALEATORIC_VARIANCE),
        )


class _AffineTarget:
    """Inverse target scaling ``y -> 2 y + 1``."""

    def inverse_transform(self, x):
        return 2.0 * np.asarray(x) + 1.0


class _IdentityFeatures:
    def transform(self, x):
        return np.asarray(x)


def _test_histories() -> pd.DataFrame:
    frames = []
    for case_id, (policy, length) in enumerate([(0, 60), (1, 45), (0, 52)]):
        time = 5.0 * np.arange(length)
        frames.append(
            pd.DataFrame(
                {
                    "run_id": case_id,
                    "policy_id": policy,
                    "time": time,
                    "load": np.linspace(0.0, 1.0, length) + case_id,
                    "soc": np.linspace(1.0, 0.1, length),
                    "voltage": np.linspace(4.2, 3.0, length),
                    "rul": time[-1] - time,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=0)


@pytest.fixture(scope="module")
def driver_inputs():
    histories = _test_histories()
    rng = np.random.default_rng(0)
    reference = {
        f"test_case_{case_id}": rng.uniform(0, 300, (N_TIMESTAMPS, N_REFERENCE))
        for case_id in histories["run_id"].unique()
    }
    return histories, reference


def _evaluate(driver_inputs, key=0, num_samples=100):
    histories, reference = driver_inputs
    return evaluate_predictive_model(
        _OffsetModel(),
        test_histories=histories,
        reference_samples=reference,
        x_scaler=_IdentityFeatures(),
        y_scaler=_AffineTarget(),
        parameters=EvaluationParameters(
            num_samples=num_samples,
            n_timestamps=N_TIMESTAMPS,
            coverage_levels=(0.5, 0.95),
        ),
        key=jax.random.key(key),
    )


def test_every_evaluated_timestamp_gets_one_noise_sample_per_draw(
    driver_inputs,
) -> None:
    evaluation = _evaluate(driver_inputs)

    assert [p.samples.name for p in evaluation.predictions] == [
        "case_0",
        "case_1",
        "case_2",
    ]
    assert [p.samples.policy for p in evaluation.predictions] == [0, 1, 0]
    for prediction in evaluation.predictions:
        samples = prediction.samples
        assert samples.predictive_samples.shape == (N_TIMESTAMPS, 100)
        assert np.isnan(samples.predictive_samples[0]).all()
        assert np.isfinite(samples.predictive_samples[1:]).all()
        assert np.all(np.diff(samples.times) > 0)
        # epistemic/aleatoric split in RUL units (scale factor 2 -> variance x4)
        np.testing.assert_allclose(
            prediction.epistemic_variance,
            4 * np.var(np.resize(OFFSETS, 100)),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            prediction.aleatoric_variance, 4 * ALEATORIC_VARIANCE, rtol=1e-5
        )


def test_the_same_key_gives_the_same_samples_and_cases_get_their_own_noise(
    driver_inputs,
) -> None:
    first, second = _evaluate(driver_inputs), _evaluate(driver_inputs)
    other = _evaluate(driver_inputs, key=1)

    def noise(evaluation, case):
        samples = evaluation.predictions[case].samples.predictive_samples[1:]
        return samples - samples.mean(axis=1, keepdims=True)

    np.testing.assert_array_equal(noise(first, 0), noise(second, 0))
    assert not np.allclose(noise(first, 0), noise(other, 0))
    assert not np.allclose(noise(first, 0)[:, :10], noise(first, 2)[:, :10])


def test_the_table_and_aggregate_follow_the_samples(driver_inputs) -> None:
    evaluation = _evaluate(driver_inputs)

    assert evaluation.table == metric_table(
        [p.samples for p in evaluation.predictions], coverage_levels=(0.5, 0.95)
    )
    assert evaluation.aggregate == aggregate(
        evaluation.table, {"case_0": 0, "case_1": 1, "case_2": 0}
    )


def test_the_evaluation_is_logged_per_case_overall_and_per_policy(
    driver_inputs, tracker_factory
) -> None:
    evaluation = _evaluate(driver_inputs)
    tracker = tracker_factory()

    log_evaluation(evaluation, tracker)

    logged = {
        key: value for values, _ in tracker.metrics for key, value in values.items()
    }
    assert logged["evaluation/test/case_1/crps"] == evaluation.table["case_1"]["crps"]
    assert logged["evaluation/test/rmse"] == evaluation.aggregate.overall["rmse"]
    assert (
        logged["evaluation/test/policy_0/coverage_95"]
        == evaluation.aggregate.per_policy[0]["coverage_95"]
    )
    assert not any(key.endswith("_average") for key in logged)
    artifact = tracker.artifacts["evaluation/test/case_2/predictive"]
    assert len(artifact["predictive_samples"]) == N_TIMESTAMPS - 1
    assert len(artifact["predictive_samples"][0]) == 100
    assert set(artifact) >= {"epistemic_variance", "aleatoric_variance", "times"}


def test_every_case_gets_a_rul_figure_next_to_its_predictive_samples(
    driver_inputs, tracker_factory
) -> None:
    evaluation = _evaluate(driver_inputs)
    tracker = tracker_factory()

    log_evaluation(evaluation, tracker)

    for prediction in evaluation.predictions:
        prefix = f"evaluation/test/{prediction.samples.name}"
        assert f"{prefix}/predictive" in tracker.artifacts
        figure = tracker.artifacts[f"{prefix}/rul_over_time"]
        assert isinstance(figure, Figure)
        assert figure.axes[0].get_title().startswith(prediction.samples.name)


@pytest.mark.parametrize(
    ("n_rows", "n_timestamps", "expected"),
    [
        (10, 4, [0, 3, 6, 9]),
        (7, 3, [0, 3, 6]),
        (5, 5, [0, 1, 2, 3, 4]),
        (3, 5, [0, 0, 1, 1, 2]),
    ],
)
def test_the_evaluation_indices_are_the_pinned_linspace_rows(
    n_rows: int, n_timestamps: int, expected: list[int]
) -> None:
    # Pins test-set scoring: values (duplicates kept when n_timestamps > n_rows)
    # and the int32 dtype.
    indices = evaluation_indices(n_rows, n_timestamps)

    assert indices.dtype == np.int32
    assert indices.tolist() == expected
