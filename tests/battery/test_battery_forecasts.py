"""Seam: interval forecasts of one test case (the reference samples)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qmodem.battery.forecasts import IntervalForecast, ReferenceSamples

_SAMPLES = st.tuples(st.integers(1, 6), st.integers(1, 20)).flatmap(
    lambda shape: arrays(
        np.float64, shape, elements=st.floats(-1e3, 1e3, allow_nan=False)
    )
)
_LEVELS = st.floats(0.01, 0.99)


@settings(max_examples=50, deadline=None)
@given(samples=_SAMPLES, level=_LEVELS)
def test_the_reference_interval_is_the_empirical_central_interval(
    samples: np.ndarray, level: float
) -> None:
    lower, upper = ReferenceSamples(samples).interval(level)

    # The empirical p-percentile (linear interpolation) lies between the order
    # statistics of rank floor(h) and ceil(h), h = (R - 1) p, 0-based.
    tail = (1 - level) / 2
    ordered = np.sort(samples, axis=1)
    n_ref = samples.shape[1]
    assert lower.shape == upper.shape == (samples.shape[0],)
    for edge, p in ((lower, tail), (upper, 1 - tail)):
        h = (n_ref - 1) * p
        below = ordered[:, int(np.floor(h))]
        above = ordered[:, int(np.ceil(h))]
        assert np.all(below - 1e-9 <= edge)
        assert np.all(edge <= above + 1e-9)


def test_the_reference_interval_interpolates_the_percentiles() -> None:
    # 0, 1, ..., 100: the p-th percentile is p itself.
    samples = np.arange(101.0)[None, ::-1]

    lower, upper = ReferenceSamples(samples).interval(0.9)

    np.testing.assert_allclose(lower, [5.0])
    np.testing.assert_allclose(upper, [95.0])


@settings(max_examples=50, deadline=None)
@given(samples=_SAMPLES, low=_LEVELS, high=_LEVELS)
def test_reference_intervals_are_nested_in_the_level(
    samples: np.ndarray, low: float, high: float
) -> None:
    low, high = sorted((low, high))
    reference = ReferenceSamples(samples)
    narrow_lower, narrow_upper = reference.interval(low)
    wide_lower, wide_upper = reference.interval(high)

    assert np.all(wide_lower <= narrow_lower)
    assert np.all(narrow_upper <= wide_upper)
    assert np.all(narrow_lower <= narrow_upper)


@settings(max_examples=25, deadline=None)
@given(samples=_SAMPLES, level=_LEVELS, seed=st.integers(0, 2**32 - 1))
def test_the_reference_interval_is_invariant_to_permuting_samples(
    samples: np.ndarray, level: float, seed: int
) -> None:
    permuted = np.random.default_rng(seed).permuted(samples, axis=1)

    for got, want in zip(
        ReferenceSamples(permuted).interval(level),
        ReferenceSamples(samples).interval(level),
        strict=True,
    ):
        np.testing.assert_array_equal(got, want)


def test_reference_samples_are_an_interval_forecast() -> None:
    assert isinstance(ReferenceSamples(np.zeros((2, 3))), IntervalForecast)
