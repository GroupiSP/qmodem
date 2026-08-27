from __future__ import annotations

import pathlib
from collections.abc import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, model_validator

from qmodem.battery.scoring import TestCaseResults


class HPOHyperparameters(BaseModel):
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

    @model_validator(mode="after")
    def validate_search_bounds(self) -> HPOHyperparameters:
        """Validates that lower search-space bounds are strictly less than upper
        bounds."""
        bounds = (
            ("window_size", self.window_size_min, self.window_size_max),
            ("kernel_size", self.kernel_size_min, self.kernel_size_ceil),
            ("conv_n_filters", self.conv_n_filters_min, self.conv_n_filters_max),
            ("beta_nll", self.beta_nll_min, self.beta_nll_max),
            ("lr", self.lr_min, self.lr_max),
            ("dropout_rate", self.dropout_rate_min, self.dropout_rate_max),
        )
        for name, lower, upper in bounds:
            if lower >= upper:
                raise ValueError(
                    f"Invalid bounds for '{name}': lower bound ({lower}) must be strictly less than upper bound ({upper})."
                )
        return self


class QAVIHPOHyperparameters(HPOHyperparameters):
    """Hyperparameters for QAVI HPO, extending the base HPO hyperparameters.

    Adds bounds for QAVI-specific search dimensions: number of qubits, number of
    PQC layers, generator and discriminator learning rates, and adversarial loss
    weight.
    """

    pqc_n_qubits_min: int = 3
    pqc_n_qubits_max: int = 8
    pqc_n_layers_min: int = 1
    pqc_n_layers_max: int = 6
    lr_generator_min: float = 1e-4
    lr_generator_max: float = 1e-2
    lr_discriminator_min: float = 1e-4
    lr_discriminator_max: float = 1e-2
    adversarial_loss_weight_min: float = 0.0
    adversarial_loss_weight_max: float = 1.0

    @model_validator(mode="after")
    def validate_qavi_search_bounds(self) -> QAVIHPOHyperparameters:
        """Validates that lower search-space bounds are strictly less than upper
        bounds."""
        bounds = (
            ("pqc_n_qubits", self.pqc_n_qubits_min, self.pqc_n_qubits_max),
            ("pqc_n_layers", self.pqc_n_layers_min, self.pqc_n_layers_max),
            ("lr_generator", self.lr_generator_min, self.lr_generator_max),
            (
                "lr_discriminator",
                self.lr_discriminator_min,
                self.lr_discriminator_max,
            ),
            (
                "adversarial_loss_weight",
                self.adversarial_loss_weight_min,
                self.adversarial_loss_weight_max,
            ),
        )
        for name, lower, upper in bounds:
            if lower >= upper:
                raise ValueError(
                    f"Invalid bounds for '{name}': lower bound ({lower}) must be strictly less than upper bound ({upper})."
                )
        return self


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
