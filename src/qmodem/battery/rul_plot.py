"""RUL-over-time figure of one evaluated test case.

The figure shows the realized RUL and the simulator's reference band over every
timestamp, and the predictive mean and band from the first evaluated timestamp on
(t0 has no prediction). Both bands are ``IntervalForecast`` intervals: the
reference band is the empirical ``ReferenceSamples`` interval; the predictive line is
the exact ``CasePredictive`` mean and its band the exact central mixture interval,
the same interval the coverage metric of ``qmodem.battery.evaluate`` scores.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .evaluate import EvaluatedCase

TARGET_LABEL = "True RUL"
REFERENCE_LABEL = "Reference RUL"
PREDICTED_LABEL = "Predicted RUL"


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
    ax.fill_between(
        case.times,
        *case.reference.interval(level),
        color="tab:blue",
        alpha=0.25,
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
    ax.fill_between(
        times,
        *case.predictive.interval(level),
        color="tab:orange",
        alpha=0.25,
        label=f"{PREDICTED_LABEL} {band}",
    )

    ax.set_title(f"{case.name} (policy {case.policy})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RUL (s)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    return figure
