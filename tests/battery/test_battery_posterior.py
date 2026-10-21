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

from qmodem.battery.posterior import (
    PosteriorKeys,
    draws_per_second,
    parameter_count,
    posterior_loader,
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
