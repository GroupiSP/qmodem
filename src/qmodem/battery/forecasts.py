"""Interval forecasts of one test case, one row per timestamp.

An ``IntervalForecast`` is any distribution over RUL, per timestamp, that yields a
central interval at a nominal level. The coverage metric, WSU and the RUL-over-time
figure consume intervals through this interface, whichever distribution they come
from. ``ReferenceSamples`` is the simulator's reference distribution, given as
samples.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IntervalForecast(Protocol):
    """A per-timestamp distribution that yields central intervals."""

    def interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the central interval at ``level`` of each timestamp.

        Args:
            level: Nominal coverage level in ``(0, 1)``.

        Returns:
            The lower and upper interval edges, one per timestamp each.
        """
        ...


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


@dataclasses.dataclass(frozen=True)
class ReferenceSamples:
    """The simulator reference RUL samples of one test case, t0 included.

    Attributes:
        samples: Reference RUL samples ``(T, R)``, row 0 at t0.
    """

    samples: np.ndarray

    def interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the empirical central interval of each timestamp.

        Args:
            level: Nominal coverage level in ``(0, 1)``.

        Returns:
            The lower and upper empirical percentiles ``(T,)`` each.
        """
        return central_band(self.samples, level)
