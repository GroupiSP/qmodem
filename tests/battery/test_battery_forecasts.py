"""Seam: interval forecasts of one test case (reference samples, posterior
predictive)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import special
from stagereg.core import expected_gaussian_mixture_crps

from qmodem.battery.forecasts import CasePredictive, IntervalForecast, ReferenceSamples

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


# --- The posterior predictive of one test case -----------------------------------

_MEANS = st.tuples(st.integers(1, 5), st.integers(1, 8)).flatmap(
    lambda shape: arrays(
        np.float64, shape, elements=st.floats(-100.0, 100.0, allow_nan=False)
    )
)


@st.composite
def case_predictives(draw: st.DrawFn) -> CasePredictive:
    """A posterior predictive with 1-5 timestamps and 1-8 draws."""
    means = draw(_MEANS)
    variance = draw(
        arrays(
            np.float64,
            means.shape[:1],
            elements=st.floats(0.01, 100.0, allow_nan=False),
        )
    )
    return CasePredictive(draw_means=means, aleatoric_variance=variance)


@settings(max_examples=50, deadline=None)
@given(predictive=case_predictives())
def test_the_predictive_mean_is_the_average_draw_mean(
    predictive: CasePredictive,
) -> None:
    np.testing.assert_allclose(
        predictive.mean(), predictive.draw_means.mean(axis=1), rtol=1e-12
    )


@settings(max_examples=50, deadline=None)
@given(predictive=case_predictives(), level=_LEVELS)
def test_the_predictive_interval_edges_are_the_central_mixture_quantiles(
    predictive: CasePredictive, level: float
) -> None:
    lower, upper = predictive.interval(level)

    # The mixture CDF, (1/M) sum_m Phi((x - mu_m) / s), hits the tail probabilities.
    def cdf(x: np.ndarray) -> np.ndarray:
        z = (x[:, None] - predictive.draw_means) / np.sqrt(
            predictive.aleatoric_variance
        )[:, None]
        return np.mean(0.5 * (1 + special.erf(z / np.sqrt(2))), axis=1)

    tail = (1 - level) / 2
    np.testing.assert_allclose(cdf(lower), tail, atol=1e-9)
    np.testing.assert_allclose(cdf(upper), 1 - tail, atol=1e-9)


@settings(max_examples=50, deadline=None)
@given(predictive=case_predictives(), low=_LEVELS, high=_LEVELS)
def test_predictive_intervals_are_nested_in_the_level(
    predictive: CasePredictive, low: float, high: float
) -> None:
    low, high = sorted((low, high))
    narrow_lower, narrow_upper = predictive.interval(low)
    wide_lower, wide_upper = predictive.interval(high)

    assert np.all(wide_lower <= narrow_lower + 1e-9)
    assert np.all(narrow_upper <= wide_upper + 1e-9)
    assert np.all(narrow_lower <= narrow_upper + 1e-9)


@settings(max_examples=50, deadline=None)
@given(
    predictive=case_predictives(),
    level=_LEVELS,
    scale=st.floats(0.1, 10.0),
    shift=st.floats(-1e3, 1e3),
)
def test_the_predictive_is_equivariant_under_an_affine_map(
    predictive: CasePredictive, level: float, scale: float, shift: float
) -> None:
    mapped = predictive.affine(scale, shift)

    np.testing.assert_allclose(mapped.draw_means, scale * predictive.draw_means + shift)
    np.testing.assert_allclose(
        mapped.aleatoric_variance, scale**2 * predictive.aleatoric_variance
    )
    np.testing.assert_allclose(
        mapped.mean(), scale * predictive.mean() + shift, rtol=1e-9, atol=1e-9
    )
    for got, want in zip(
        mapped.interval(level), predictive.interval(level), strict=True
    ):
        np.testing.assert_allclose(got, scale * want + shift, rtol=1e-7, atol=1e-6)


@settings(max_examples=50, deadline=None)
@given(predictive=case_predictives(), data=st.data())
def test_the_predictive_crps_is_the_mixture_crps_without_the_t0_reference_row(
    predictive: CasePredictive, data
) -> None:
    n_ref = data.draw(st.integers(1, 6))
    reference = data.draw(
        arrays(
            np.float64,
            (predictive.draw_means.shape[0] + 1, n_ref),
            elements=st.floats(-100.0, 100.0, allow_nan=False),
        )
    )
    reference[0] = np.nan  # t0 is never scored

    np.testing.assert_allclose(
        predictive.crps(ReferenceSamples(reference)),
        expected_gaussian_mixture_crps(
            predictive.draw_means, predictive.aleatoric_variance, reference[1:]
        ),
    )


@settings(max_examples=25, deadline=None)
@given(predictive=case_predictives(), level=_LEVELS, seed=st.integers(0, 2**32 - 1))
def test_the_predictive_is_invariant_to_permuting_draws(
    predictive: CasePredictive, level: float, seed: int
) -> None:
    permuted = CasePredictive(
        draw_means=np.random.default_rng(seed).permuted(predictive.draw_means, axis=1),
        aleatoric_variance=predictive.aleatoric_variance,
    )

    np.testing.assert_allclose(permuted.mean(), predictive.mean(), atol=1e-9)
    for got, want in zip(
        permuted.interval(level), predictive.interval(level), strict=True
    ):
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_the_case_predictive_is_an_interval_forecast() -> None:
    predictive = CasePredictive(
        draw_means=np.zeros((2, 3)), aleatoric_variance=np.ones(2)
    )

    assert isinstance(predictive, IntervalForecast)
