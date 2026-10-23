"""The shared posterior stage: its keys and plug-in pieces."""

from __future__ import annotations

import dataclasses
import itertools
import statistics

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from stagereg.core import (
    SampleContinuousRankedProbabilityScore,
    expected_gaussian_mixture_crps,
)
from stagereg.jax import MetricSuite, PosteriorPredictive, PosteriorPredictivePredictor

from qmodem.battery.posterior import (
    PosteriorKeys,
    diagnostic_metrics,
    draws_per_second,
    parameter_count,
    posterior_loader,
    score_validation,
    validation_crps,
    validation_metrics,
)


def test_the_posterior_stage_validates_on_the_posterior_validation_set(
    synthetic_battery,
) -> None:
    data = synthetic_battery.data
    loader = posterior_loader(data, batch_size=8, key=jax.random.key(0))

    (batch,) = loader(split="validation", epoch=0)

    np.testing.assert_array_equal(
        np.asarray(batch.inputs), np.asarray(data.posterior_val.features)
    )
    np.testing.assert_array_equal(
        np.asarray(batch.targets), np.asarray(data.posterior_val.targets)
    )


def _bits(key: jax.Array) -> tuple[int, ...]:
    return tuple(np.asarray(jax.random.key_data(key)).ravel().tolist())


@given(seed=st.integers(0, 2**31 - 1))
def test_posterior_keys_are_reproducible_and_distinct(seed: int) -> None:
    root = jax.random.key(seed)
    keys = PosteriorKeys.of(root)
    fields = [_bits(getattr(keys, f.name)) for f in dataclasses.fields(keys)]

    assert fields == [
        _bits(getattr(PosteriorKeys.of(root), f.name)) for f in dataclasses.fields(keys)
    ]
    assert len({*fields, _bits(root)}) == len(fields) + 1


@given(
    shapes=st.lists(
        st.lists(st.integers(1, 4), min_size=0, max_size=3), min_size=0, max_size=4
    )
)
def test_the_parameter_count_is_the_number_of_entries(shapes: list[list[int]]) -> None:
    tree = {str(i): jnp.zeros(shape) for i, shape in enumerate(shapes)}

    assert parameter_count(tree) == sum(int(np.prod(shape)) for shape in shapes)


def _draw(inputs: jax.Array, key: jax.Array) -> jax.Array:
    return inputs + jax.random.normal(key, inputs.shape)


_INPUTS = jnp.zeros(1)
_KEY = jax.random.key(0)


def _clock(durations: list[float]):
    """A clock whose consecutive (start, end) readings span ``durations``."""
    readings = itertools.chain.from_iterable(
        (float(i), i + duration) for i, duration in enumerate(durations)
    )
    return lambda: next(readings)


@settings(deadline=None)  # the first example compiles the draw
@given(
    durations=st.lists(st.floats(1e-3, 1e3), min_size=1, max_size=9),
    num_draws=st.integers(1, 1000),
)
def test_draws_per_second_is_the_draws_over_the_median_call_time(
    durations: list[float], num_draws: int
) -> None:
    rate = draws_per_second(
        _draw,
        _INPUTS,
        _KEY,
        num_draws,
        repeats=len(durations),
        clock=_clock(durations),
    )

    assert rate == pytest.approx(num_draws / statistics.median(durations))


@settings(deadline=None)
@given(outlier=st.floats(1e-3, 1e3))
def test_draws_per_second_ignores_one_outlying_call(outlier: float) -> None:
    durations = [1.0, 1.0, outlier]

    rate = draws_per_second(
        _draw, _INPUTS, _KEY, 10, repeats=3, clock=_clock(durations)
    )

    assert rate == pytest.approx(10.0)


@pytest.mark.parametrize("repeats", [0, -1])
def test_draws_per_second_needs_a_positive_repeat_count(repeats: int) -> None:
    with pytest.raises(ValueError, match="repeats"):
        draws_per_second(_draw, _INPUTS, _KEY, 1, repeats=repeats)


def test_draws_per_second_passes_inputs_and_key_at_runtime() -> None:
    traced = []

    def draw(inputs: jax.Array, key: jax.Array) -> jax.Array:
        traced.append(
            isinstance(inputs, jax.core.Tracer) and isinstance(key, jax.core.Tracer)
        )
        return _draw(inputs, key)

    draws_per_second(draw, _INPUTS, _KEY, 1, repeats=1)

    assert traced == [False, True]  # one eager call, one trace


# --- the validation criterion: closed-form CRPS of the posterior predictive -------


def _toy_predictive(num_draws: int, variance: float):
    """A posterior predictive whose draws depend on the inputs and the draw key."""

    def predictive(inputs: jax.Array, key: jax.Array) -> PosteriorPredictive:
        location = jnp.mean(inputs, axis=tuple(range(1, inputs.ndim)))[:, None]
        offsets = jax.random.normal(key, (num_draws, 1, 1))
        return PosteriorPredictive(
            draws=location[None] + offsets,
            aleatoric_variance=jnp.full(location.shape, variance),
        )

    return predictive


@settings(max_examples=10, deadline=None)
@given(
    seed=st.integers(0, 2**16),
    num_draws=st.integers(1, 8),
    variance=st.floats(1e-2, 4.0),
    batch_size=st.integers(1, 64),
)
def test_validation_crps_is_the_closed_form_mixture_crps_of_the_validation_targets(
    synthetic_battery, seed: int, num_draws: int, variance: float, batch_size: int
) -> None:
    data = synthetic_battery.data
    predictive = _toy_predictive(num_draws, variance)
    validation_key = jax.random.key(seed)
    loader = posterior_loader(data, batch_size=batch_size, key=jax.random.key(0))

    crps = validation_crps(predictive, loader, validation_key)

    # Only the draw key of stagereg's split selects the draws (ADR 0002).
    draw_key, _ = jax.random.split(validation_key)
    features = data.posterior_val.features
    targets = np.asarray(data.posterior_val.targets).reshape(-1, 1)
    mixture = predictive(features, draw_key)
    expected = expected_gaussian_mixture_crps(
        np.asarray(mixture.draws)[..., 0].T,
        np.asarray(mixture.aleatoric_variance)[:, 0],
        targets,
    )
    assert crps == pytest.approx(float(np.mean(expected)), rel=1e-5)


@pytest.mark.parametrize("suite", [validation_metrics, diagnostic_metrics])
def test_no_validation_suite_requests_predictive_samples(suite) -> None:
    metrics = suite()

    assert metrics.predictive_sample_count is None
    assert not metrics.prediction_requirements.fields.predictive_samples


def test_scoring_a_suite_that_draws_predictive_samples_is_refused(
    synthetic_battery,
) -> None:
    sampled = MetricSuite(
        epoch=(SampleContinuousRankedProbabilityScore(name="crps"),),
        predictor=PosteriorPredictivePredictor(),
        predictive_sample_count=10,
    )
    loader = posterior_loader(
        synthetic_battery.data, batch_size=8, key=jax.random.key(0)
    )

    with pytest.raises(ValueError, match="predictive samples"):
        score_validation(_toy_predictive(2, 1.0), loader, jax.random.key(0), sampled)
