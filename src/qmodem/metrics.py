from __future__ import annotations

import numpy as np


def cdf(x: float, samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0

    sorted_samples = np.sort(samples)
    count = np.sum(sorted_samples <= x)
    cdf_value = count / len(sorted_samples)

    return cdf_value


def crps(
    samples_true: np.ndarray, samples_pred: np.ndarray, x_grid: np.ndarray
) -> float:
    F0 = np.array([cdf(x, samples_true) for x in x_grid])
    F1 = np.array([cdf(x, samples_pred) for x in x_grid])

    return np.trapezoid((F0 - F1) ** 2, x_grid)


def point_crps(
    y_true: np.ndarray, samples_pred: np.ndarray, x_grid: np.ndarray
) -> float:
    F0 = np.where(x_grid < y_true, 0.0, 1.0)
    F1 = np.array([cdf(x, samples_pred) for x in x_grid])

    return np.trapezoid((F0 - F1) ** 2, x_grid)
