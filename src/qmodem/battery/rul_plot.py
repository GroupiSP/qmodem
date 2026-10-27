"""RUL-over-time figure of one evaluated test case.

The figure shows the realized RUL and the simulator's reference band over every
timestamp, and the predictive mean and band from the first evaluated timestamp on
(t0 has no prediction). Both bands are ``IntervalForecast`` intervals, drawn by
``fill_interval``: the reference band is the empirical ``ReferenceSamples``
interval; the predictive line is the exact ``CasePredictive`` mean and its band the
exact central mixture interval, the same interval the coverage metric of
``qmodem.battery.evaluate`` scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .forecasts import IntervalForecast

if TYPE_CHECKING:
    from .evaluate import EvaluatedCase

TARGET_LABEL = "True RUL"
REFERENCE_LABEL = "Reference RUL"
PREDICTED_LABEL = "Predicted RUL"


def fill_interval(
    ax: Axes,
    times: np.ndarray,
    forecast: IntervalForecast,
    level: float,
    *,
    color: str,
    label: str,
) -> None:
    """Shade the central interval of a forecast over time.

    Args:
        ax: The axes to draw on.
        times: Timestamp of each forecast row.
        forecast: Any interval forecast, one row per timestamp.
        level: Nominal level of the interval.
        color: Fill colour.
        label: Legend label.
    """
    ax.fill_between(
        times, *forecast.interval(level), color=color, alpha=0.25, label=label
    )


def plot_rul_over_time(case: EvaluatedCase, *, level: float) -> Figure:
    """Plot the realized, reference and predicted RUL of one test case over time.

    The figure is built without ``pyplot``, so it is not registered with a GUI
    backend and needs no explicit closing.

    Args:
        case: The evaluated test case (row 0 is t0, without prediction).
        level: Nominal level of the reference and predictive bands.

    Returns:
        A figure with a single axes.
    """
    figure = Figure(figsize=(6.0, 4.0), layout="constrained")
    ax = figure.add_subplot()
    band = f"{100 * level:g}% band"

    ax.plot(case.times, case.targets, color="black", label=TARGET_LABEL)
    fill_interval(
        ax,
        case.times,
        case.reference,
        level,
        color="tab:blue",
        label=f"{REFERENCE_LABEL} {band}",
    )
    times = case.times[1:]
    ax.plot(
        times,
        case.predictive.mean(),
        "-o",
        color="tab:orange",
        markersize=3,
        label=PREDICTED_LABEL,
    )
    fill_interval(
        ax,
        times,
        case.predictive,
        level,
        color="tab:orange",
        label=f"{PREDICTED_LABEL} {band}",
    )

    ax.set_title(f"{case.name} (policy {case.policy})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RUL (s)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    return figure
