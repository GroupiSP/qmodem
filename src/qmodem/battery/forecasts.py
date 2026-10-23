"""Interval forecasts of one test case, one row per timestamp.

An ``IntervalForecast`` is any distribution over RUL, per timestamp, that yields a
central interval at a nominal level. The coverage metric, WSU and the RUL-over-time
figure consume intervals through this interface, whichever distribution they come
from. ``ReferenceSamples`` is the simulator's reference distribution, given as
samples; its interval is empirical. ``CasePredictive`` is a model's posterior
predictive, the exact mixture ``(1/M) sum_m N(mu_m, s**2)``; its interval is made of
exact mixture quantiles.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import numpy as np
from scipy import special
from stagereg.core import expected_gaussian_mixture_crps

_BISECTION_STEPS = 100
"""Halvings of the quantile bracket, past float64 resolution for RUL brackets."""


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
        tail = 100 * (1 - level) / 2
        lower, upper = np.percentile(self.samples, [tail, 100 - tail], axis=1)
        return lower, upper


@dataclasses.dataclass(frozen=True)
class CasePredictive:
    """The posterior predictive of one test case, t0 excluded.

    At each evaluated timestamp it is the equally weighted mixture
    ``(1/M) sum_m N(mu_m, s**2)`` of the posterior draws' Gaussians, all sharing the
    timestamp's aleatoric variance ``s**2``. Every quantity is computed exactly from
    it; nothing is sampled.

    Attributes:
        draw_means: Mean of each posterior draw ``(T - 1, M)``.
        aleatoric_variance: The variance every draw shares ``(T - 1,)``.
    """

    draw_means: np.ndarray
    aleatoric_variance: np.ndarray

    def affine(self, scale: float, shift: float) -> CasePredictive:
        """Return the predictive of ``scale * Y + shift``.

        Each draw mean maps to ``scale * mu_m + shift``, the variance to
        ``scale**2 * s**2``.

        Args:
            scale: The affine scale ``a``.
            shift: The affine shift ``b``.

        Returns:
            The mapped predictive.
        """
        return CasePredictive(
            draw_means=scale * self.draw_means + shift,
            aleatoric_variance=scale**2 * self.aleatoric_variance,
        )

    def mean(self) -> np.ndarray:
        """Return the exact mixture mean, the average draw mean.

        Returns:
            The mean of each evaluated timestamp ``(T - 1,)``.
        """
        return self.draw_means.mean(axis=1)

    def interval(self, level: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact central mixture interval of each timestamp.

        The edges are the mixture quantiles at ``(1 -/+ level) / 2``, found in
        double precision as stagereg's ``PosteriorPredictive.quantile`` finds
        them (``_mixture_quantile``).

        Args:
            level: Nominal coverage level in ``(0, 1)``.

        Returns:
            The lower and upper mixture quantiles ``(T - 1,)`` each.

        Raises:
            ValueError: If ``level`` is not strictly between 0 and 1.
        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must be strictly between 0 and 1.")
        lower = _mixture_quantile(
            self.draw_means, self.aleatoric_variance, (1.0 - level) / 2.0
        )
        upper = _mixture_quantile(
            self.draw_means, self.aleatoric_variance, (1.0 + level) / 2.0
        )
        return lower, upper

    def crps(self, reference: ReferenceSamples) -> np.ndarray:
        """Return the closed-form expected CRPS against the reference samples.

        Args:
            reference: The case's reference samples, t0 included; the t0 row is
                dropped.

        Returns:
            The expected CRPS of each evaluated timestamp ``(T - 1,)``.

        Raises:
            ValueError: If the reference does not have one row more than the
                predictive, or an input is not finite (see
                ``stagereg.core.expected_gaussian_mixture_crps``).
        """
        return expected_gaussian_mixture_crps(
            self.draw_means, self.aleatoric_variance, reference.samples[1:]
        )


def _mixture_quantile(
    draw_means: np.ndarray, variance: np.ndarray, probability: float
) -> np.ndarray:
    """Invert the mixture CDF ``(1/M) sum_m Phi((x - mu_m) / s)`` per timestamp.

    It is stagereg's ``PosteriorPredictive.quantile`` algorithm in NumPy float64,
    so it needs no JAX double-precision mode: the draws' Gaussian quantiles
    ``mu_m + s * Phi^-1(p)`` bracket the mixture quantile, and the bracket is
    halved ``_BISECTION_STEPS`` times.

    Args:
        draw_means: Mean of each draw ``(T, M)``.
        variance: The variance every draw shares ``(T,)``.
        probability: The cumulative probability, strictly between 0 and 1.

    Returns:
        The mixture quantile of each timestamp ``(T,)``.
    """
    means = np.asarray(draw_means, dtype=np.float64)
    scale = np.sqrt(np.asarray(variance, dtype=np.float64))[:, None]
    component = means + special.ndtri(probability) * scale
    low, high = component.min(axis=1), component.max(axis=1)
    for _ in range(_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        below = special.ndtr((middle[:, None] - means) / scale).mean(axis=1)
        below = below < probability
        low, high = np.where(below, middle, low), np.where(below, high, middle)
    return 0.5 * (low + high)
