"""Seam 3: the evaluation driver (posterior predictive and reference samples per
test case -> metric table and aggregates, plus the prediction and logging around
that core)."""

from __future__ import annotations

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from matplotlib.figure import Figure
from stagereg.core import expected_gaussian_mixture_crps, wsu
from stagereg.jax import PosteriorPredictive

from qmodem.battery.evaluate import (
    EvaluatedCase,
    Evaluation,
    EvaluationParameters,
    aggregate,
    case_metrics,
    evaluate_predictive_model,
    evaluation_indices,
    log_evaluation,
    metric_table,
)
from qmodem.battery.forecasts import CasePredictive, ReferenceSamples

_FINITE = st.floats(-50.0, 50.0, allow_nan=False, width=32)


@st.composite
def evaluated_cases(draw: st.DrawFn, *, name: str = "case_0", policy: int = 0):
    """A test case with t0 plus at least two evaluated timestamps."""
    n_times = draw(st.integers(3, 7))
    n_draws = draw(st.integers(1, 12))
    n_ref = draw(st.integers(2, 12))
    steps = draw(st.lists(st.floats(0.5, 20.0), min_size=n_times, max_size=n_times))
    arrays = st.lists(_FINITE, min_size=1, max_size=1).map(lambda v: v[0])

    def matrix(rows: int, cols: int) -> np.ndarray:
        values = draw(st.lists(arrays, min_size=rows * cols, max_size=rows * cols))
        return np.asarray(values, dtype=np.float64).reshape(rows, cols)

    variance = draw(
        st.lists(st.floats(0.01, 25.0), min_size=n_times - 1, max_size=n_times - 1)
    )
    return EvaluatedCase(
        name=name,
        policy=policy,
        times=np.cumsum(steps),
        targets=matrix(n_times, 1)[:, 0],
        reference=ReferenceSamples(matrix(n_times, n_ref)),
        predictive=CasePredictive(
            draw_means=matrix(n_times - 1, n_draws),
            aleatoric_variance=np.asarray(variance),
        ),
    )


def _assert_metrics_close(actual: dict, expected: dict, tol: float = 1e-9) -> None:
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key] == pytest.approx(expected[key], rel=tol, abs=tol), key


@settings(max_examples=50, deadline=None)
@given(case=evaluated_cases(), t0_values=st.data())
def test_the_t0_row_never_changes_a_metric(case: EvaluatedCase, t0_values) -> None:
    other = t0_values.draw(evaluated_cases())
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
    )

    assert case_metrics(replaced, coverage_levels=(0.5, 0.95)) == case_metrics(
        case, coverage_levels=(0.5, 0.95)
    )


@settings(max_examples=50, deadline=None)
@given(case=evaluated_cases(), seed=st.integers(0, 2**32 - 1))
def test_metrics_are_invariant_to_permuting_draws(
    case: EvaluatedCase, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    permuted = dataclasses.replace(
        case,
        predictive=dataclasses.replace(
            case.predictive,
            draw_means=rng.permuted(case.predictive.draw_means, axis=1),
        ),
    )

    _assert_metrics_close(
        case_metrics(permuted, coverage_levels=(0.95,)),
        case_metrics(case, coverage_levels=(0.95,)),
    )


@settings(max_examples=50, deadline=None)
@given(
    case=evaluated_cases(),
    levels=st.lists(st.floats(0.01, 0.99), min_size=2, max_size=6, unique=True),
)
def test_coverage_is_monotone_non_decreasing_in_the_level(
    case: EvaluatedCase, levels: list[float]
) -> None:
    levels = sorted(levels)
    metrics = case_metrics(case, coverage_levels=levels)

    coverages = [metrics[f"coverage_{100 * level:g}"] for level in levels]
    assert coverages == sorted(coverages)


@settings(max_examples=30, deadline=None)
@given(case=evaluated_cases())
def test_crps_is_the_mean_closed_form_mixture_crps_without_t0(
    case: EvaluatedCase,
) -> None:
    expected = expected_gaussian_mixture_crps(
        case.predictive.draw_means,
        case.predictive.aleatoric_variance,
        case.reference.samples[1:],
    ).mean()

    assert case_metrics(case, coverage_levels=(0.95,))["crps"] == pytest.approx(
        expected
    )


@settings(max_examples=30, deadline=None)
@given(case=evaluated_cases())
def test_rmse_is_scored_on_the_mixture_mean(case: EvaluatedCase) -> None:
    errors = case.targets[1:] - case.predictive.draw_means.mean(axis=1)

    assert case_metrics(case, coverage_levels=(0.95,))["rmse"] == pytest.approx(
        np.sqrt(np.mean(errors**2))
    )


@settings(max_examples=30, deadline=None)
@given(case=evaluated_cases())
def test_wsu_is_stageregs_per_trajectory_wsu_of_the_mixture_interval(
    case: EvaluatedCase,
) -> None:
    lower, upper = case.predictive.interval(0.95)
    expected = wsu(
        np.concatenate([lower[:1], lower]),
        np.concatenate([upper[:1], upper]),
        case.times,
    )

    assert case_metrics(case, coverage_levels=(0.95,))["wsu"] == pytest.approx(expected)


@settings(max_examples=30, deadline=None)
@given(
    case=evaluated_cases(),
    scale=st.floats(0.1, 10.0),
    shift=st.floats(-100.0, 100.0),
)
def test_metrics_are_equivariant_under_an_affine_rul_map(
    case: EvaluatedCase, scale: float, shift: float
) -> None:
    # Mapping the targets, references and predictive by y -> a y + b scales CRPS,
    # RMSE and WSU by a, and leaves coverage unchanged.
    mapped = dataclasses.replace(
        case,
        targets=scale * case.targets + shift,
        reference=ReferenceSamples(scale * case.reference.samples + shift),
        predictive=case.predictive.affine(scale, shift),
    )
    expected = case_metrics(case, coverage_levels=(0.5,))
    actual = case_metrics(mapped, coverage_levels=(0.5,))

    for metric in ("crps", "rmse", "wsu"):
        assert actual[metric] == pytest.approx(scale * expected[metric], rel=1e-6)
    assert actual["coverage_50"] == expected["coverage_50"]


def test_rmse_and_coverage_on_a_worked_example() -> None:
    case = EvaluatedCase(
        name="case_0",
        policy=0,
        times=np.array([0.0, 10.0, 20.0, 30.0]),
        targets=np.array([99.0, 1.0, 5.0, 100.0]),
        reference=ReferenceSamples(np.zeros((4, 3))),
        predictive=CasePredictive(
            draw_means=np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]),
            aleatoric_variance=np.full(3, 1e-4),
        ),
    )

    metrics = case_metrics(case, coverage_levels=(0.95,))

    # mixture means 1, 3, 5 vs targets 1, 5, 100: squared errors 0, 4, 9025
    assert metrics["rmse"] == pytest.approx(np.sqrt((0 + 4 + 9025) / 3))
    # the 95% interval is about [mu_1, mu_2]: only the first target lies inside
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
        EvaluatedCase(
            name=f"case_{i}",
            policy=i % 2,
            times=np.array([0.0, 1.0, 2.0]),
            targets=np.array([0.0, 1.0, 2.0]),
            reference=ReferenceSamples(np.ones((3, 2))),
            predictive=CasePredictive(
                draw_means=np.ones((2, 4)) * i, aleatoric_variance=np.ones(2)
            ),
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


def _evaluate(driver_inputs, key=0, num_draws=100):
    histories, reference = driver_inputs
    return evaluate_predictive_model(
        _OffsetModel(),
        test_histories=histories,
        reference_samples=reference,
        x_scaler=_IdentityFeatures(),
        y_scaler=_AffineTarget(),
        parameters=EvaluationParameters(
            num_draws=num_draws,
            n_timestamps=N_TIMESTAMPS,
            coverage_levels=(0.5, 0.95),
        ),
        key=jax.random.key(key),
    )


@pytest.fixture(scope="module")
def evaluation(driver_inputs) -> Evaluation:
    return _evaluate(driver_inputs)


def _logged_metrics(tracker) -> dict[str, float]:
    return {
        key: value for values, _ in tracker.metrics for key, value in values.items()
    }


def test_every_evaluated_timestamp_gets_the_draws_in_rul_units(evaluation) -> None:
    assert [case.name for case in evaluation.cases] == ["case_0", "case_1", "case_2"]
    assert [case.policy for case in evaluation.cases] == [0, 1, 0]
    offsets = np.resize(OFFSETS, 100)
    for case in evaluation.cases:
        predictive = case.predictive
        assert predictive.draw_means.shape == (N_TIMESTAMPS - 1, 100)
        assert np.all(np.diff(case.times) > 0)
        # y -> 2 y + 1: the draws' offsets double and the variance quadruples.
        centred = predictive.draw_means - predictive.draw_means[:, :1]
        np.testing.assert_allclose(
            centred,
            np.tile(2 * (offsets - offsets[0]), (N_TIMESTAMPS - 1, 1)),
            atol=1e-5,
        )
        np.testing.assert_allclose(
            predictive.aleatoric_variance, 4 * ALEATORIC_VARIANCE, rtol=1e-5
        )


def test_given_the_draws_the_metrics_do_not_depend_on_the_key(
    driver_inputs, evaluation
) -> None:
    # _OffsetModel's draws ignore the key, so only sampled noise could differ.
    assert _evaluate(driver_inputs, key=1).table == evaluation.table


def test_the_table_and_aggregate_follow_the_cases(evaluation) -> None:
    assert evaluation.table == metric_table(
        evaluation.cases, coverage_levels=(0.5, 0.95)
    )
    assert evaluation.aggregate == aggregate(
        evaluation.table, {"case_0": 0, "case_1": 1, "case_2": 0}
    )


def test_the_evaluation_is_logged_per_case_overall_and_per_policy(
    evaluation, tracker_factory
) -> None:
    tracker = tracker_factory()

    log_evaluation(evaluation, tracker)

    logged = _logged_metrics(tracker)
    assert logged["evaluation/test/case_1/crps"] == evaluation.table["case_1"]["crps"]
    assert logged["evaluation/test/rmse"] == evaluation.aggregate.overall["rmse"]
    assert (
        logged["evaluation/test/policy_0/coverage_95"]
        == evaluation.aggregate.per_policy[0]["coverage_95"]
    )
    assert not any(key.endswith("_average") for key in logged)


def test_the_artifact_holds_exactly_the_posterior_predictive(
    evaluation, tracker_factory
) -> None:
    tracker = tracker_factory()

    log_evaluation(evaluation, tracker)

    artifact = tracker.artifacts["evaluation/test/case_2/predictive"]
    assert set(artifact) == {
        "name",
        "policy",
        "times",
        "targets",
        "reference_samples",
        "draw_means",
        "aleatoric_variance",
    }
    assert np.shape(artifact["draw_means"]) == (N_TIMESTAMPS - 1, 100)
    assert np.shape(artifact["aleatoric_variance"]) == (N_TIMESTAMPS - 1,)
    assert np.shape(artifact["reference_samples"]) == (N_TIMESTAMPS, N_REFERENCE)


def test_the_case_rebuilt_from_the_artifact_reproduces_every_logged_metric(
    evaluation, tracker_factory
) -> None:
    tracker = tracker_factory()

    log_evaluation(evaluation, tracker)

    logged = _logged_metrics(tracker)
    for case in evaluation.cases:
        artifact = json.loads(
            json.dumps(tracker.artifacts[f"evaluation/test/{case.name}/predictive"])
        )
        rebuilt = case_metrics(
            EvaluatedCase.from_artifact(artifact), coverage_levels=(0.5, 0.95)
        )
        for metric, value in rebuilt.items():
            assert logged[f"evaluation/test/{case.name}/{metric}"] == value


def test_every_case_gets_a_rul_figure_next_to_its_predictive(
    evaluation, tracker_factory
) -> None:
    tracker = tracker_factory()

    log_evaluation(evaluation, tracker)

    for case in evaluation.cases:
        prefix = f"evaluation/test/{case.name}"
        assert f"{prefix}/predictive" in tracker.artifacts
        figure = tracker.artifacts[f"{prefix}/rul_over_time"]
        assert isinstance(figure, Figure)
        assert figure.axes[0].get_title().startswith(case.name)


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
