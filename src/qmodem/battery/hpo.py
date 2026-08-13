from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from qmodem.battery.scoring import TestCaseResults


@dataclasses.dataclass
class HPOHyperparameters:
    """Hyperparameters for the HPO."""

    seed_objective: int = 42
    seed_hp_sampler: int = 123
    rul_grid_crps_start: float = 0.0
    rul_grid_crps_end: float = 5_000.0
    rul_grid_crps_resolution: float = 50.0
    num_mc_samples: int = 100
    num_hp_trials: int = 25
    num_soc0s_eval: int = 20
    window_size_min: int = 10
    window_size_max: int = 100
    kernel_size_min: int = 3
    kernel_size_ceil: int = 20
    conv_n_filters_min: int = 4
    conv_n_filters_max: int = 40
    beta_nll_min: float = 0.0
    beta_nll_max: float = 1.0
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    dropout_rate_min: float = 0.0
    dropout_rate_max: float = 0.9


def get_validation_data(path: pathlib.Path, ids: Sequence[int]) -> pd.DataFrame:
    """Returns the validation data from the training data."""
    df = pd.read_csv(path)
    return df[df["run_id"].isin(ids)]


def get_average_crps(
    test_case_results: Sequence[TestCaseResults], x_grid: np.ndarray
) -> float:
    """Computes the average CRPS over all test cases."""
    crps_values = []
    for test_case in test_case_results:
        crps_values.append(test_case.average_crps(x_grid))
    return np.mean(crps_values)
