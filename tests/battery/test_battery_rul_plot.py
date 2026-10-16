"""The RUL-over-time figure of one evaluated test case."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from qmodem.battery.evaluate import CaseSamples
from qmodem.battery.rul_plot import (
    PREDICTED_LABEL,
    REFERENCE_LABEL,
    TARGET_LABEL,
    central_band,
    plot_rul_over_time,
)

N_TIMES = 6


@pytest.fixture(scope="module")
def case() -> CaseSamples:
    rng = np.random.default_rng(0)
    times = 10.0 * np.arange(N_TIMES)
    targets = times[-1] - times + 5.0
    predictive = targets[:, None] + rng.normal(0.0, 3.0, (N_TIMES, 40))
    predictive[0] = np.nan  # t0 has no prediction
    return CaseSamples(
        name="case_4",
        policy=2,
        times=times,
        targets=targets,
        reference_samples=targets[:, None] + rng.normal(0.0, 5.0, (N_TIMES, 30)),
        predictive_samples=predictive,
    )


def _axes(figure: Figure) -> Axes:
    (ax,) = figure.axes
    return ax


def _line(ax: Axes, label: str) -> tuple[np.ndarray, np.ndarray]:
    (line,) = [line for line in ax.get_lines() if line.get_label() == label]
    return np.asarray(line.get_xdata()), np.asarray(line.get_ydata())


def _band_extent(ax: Axes, label: str) -> tuple[float, float]:
    (band,) = [
        c
        for c in ax.collections
        if isinstance(c, PolyCollection) and c.get_label().startswith(label)
    ]
    y = band.get_paths()[0].vertices[:, 1]
    return float(y.min()), float(y.max())


def test_the_figure_has_one_titled_axes_with_a_non_negative_rul_axis(case) -> None:
    ax = _axes(plot_rul_over_time(case, level=0.95))

    assert ax.get_title() == "case_4 (policy 2)"
    assert ax.get_ylim()[0] == 0
    assert ax.get_legend() is not None


def test_the_target_spans_every_timestamp_and_the_prediction_skips_t0(case) -> None:
    ax = _axes(plot_rul_over_time(case, level=0.95))

    target_t, target_y = _line(ax, TARGET_LABEL)
    np.testing.assert_array_equal(target_t, case.times)
    np.testing.assert_array_equal(target_y, case.targets)
    predicted_t, predicted_y = _line(ax, PREDICTED_LABEL)
    np.testing.assert_array_equal(predicted_t, case.times[1:])
    np.testing.assert_allclose(predicted_y, case.predictive_samples[1:].mean(axis=1))


@pytest.mark.parametrize(
    ("label", "rows"),
    [(REFERENCE_LABEL, slice(None)), (PREDICTED_LABEL, slice(1, None))],
)
def test_the_bands_are_the_metric_central_bands(case, label, rows) -> None:
    samples = {
        REFERENCE_LABEL: case.reference_samples,
        PREDICTED_LABEL: case.predictive_samples,
    }[label][rows]
    ax = _axes(plot_rul_over_time(case, level=0.9))

    lower, upper = central_band(samples, 0.9)
    assert _band_extent(ax, label) == pytest.approx((lower.min(), upper.max()))


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_the_figure_is_invariant_to_permuting_samples_within_a_timestamp(
    case, seed
) -> None:
    rng = np.random.default_rng(seed)
    permuted = CaseSamples(
        name=case.name,
        policy=case.policy,
        times=case.times,
        targets=case.targets,
        reference_samples=rng.permuted(case.reference_samples, axis=1),
        predictive_samples=rng.permuted(case.predictive_samples, axis=1),
    )
    original = _axes(plot_rul_over_time(case, level=0.95))
    shuffled = _axes(plot_rul_over_time(permuted, level=0.95))

    np.testing.assert_allclose(
        _line(shuffled, PREDICTED_LABEL)[1], _line(original, PREDICTED_LABEL)[1]
    )
    for label in (REFERENCE_LABEL, PREDICTED_LABEL):
        assert _band_extent(shuffled, label) == pytest.approx(
            _band_extent(original, label)
        )


@settings(max_examples=25, deadline=None)
@given(
    low=st.floats(0.05, 0.9),
    gap=st.floats(0.01, 0.09),
)
def test_a_wider_level_never_narrows_the_band(case, low, gap) -> None:
    narrow = _axes(plot_rul_over_time(case, level=low))
    wide = _axes(plot_rul_over_time(case, level=low + gap))

    for label in (REFERENCE_LABEL, PREDICTED_LABEL):
        (narrow_lo, narrow_hi), (wide_lo, wide_hi) = (
            _band_extent(narrow, label),
            _band_extent(wide, label),
        )
        assert wide_lo <= narrow_lo + 1e-9
        assert wide_hi >= narrow_hi - 1e-9
