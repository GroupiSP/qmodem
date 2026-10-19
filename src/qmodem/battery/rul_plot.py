"""RUL-over-time figure of one evaluated test case.

The figure shows the realized RUL and the simulator's reference band over every
timestamp, and the predictive mean and band from the first evaluated timestamp on
(t0 has no prediction). Bands are the same empirical central bands as the coverage
metric of ``qmodem.battery.evaluate``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .evaluate import CaseSamples

TARGET_LABEL = "True RUL"
REFERENCE_LABEL = "Reference RUL"
PREDICTED_LABEL = "Predicted RUL"


def central_band(samples: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical central band of each row of ``samples``.

    Args:
        samples: Samples ``(U, M)``.
        level: Nominal coverage level in ``(0, 1)``.

    Returns:
        The lower and upper empirical percentiles ``(U,)`` each.
    """
    tail = 100 * (1 - level) / 2
    lower, upper = np.percentile(samples, [tail, 100 - tail], axis=1)
    return lower, upper


def plot_rul_over_time(case: CaseSamples, *, level: float) -> Figure:
    """Plot the realized, reference and predicted RUL of one test case over time.

    The figure is built without ``pyplot``, so it is not registered with a GUI
    backend and needs no explicit closing.

    Args:
        case: The test case's samples (row 0 is t0, without prediction).
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
        *central_band(case.reference_samples, level),
        color="tab:blue",
        alpha=0.25,
        label=f"{REFERENCE_LABEL} {band}",
    )
    times = case.times[1:]
    predictive = case.predictive_samples[1:]
    ax.plot(
        times,
        predictive.mean(axis=1),
        "-o",
        color="tab:orange",
        markersize=3,
        label=PREDICTED_LABEL,
    )
    ax.fill_between(
        times,
        *central_band(predictive, level),
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
