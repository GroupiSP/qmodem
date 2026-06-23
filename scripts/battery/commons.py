from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Iterable, Iterator

import grain
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import simbat as sb

from qmodem.data import DataSource

DATA_GEN_RUN_ID = "48ce4a61104840c58e892006c1bc7880"


@dataclasses.dataclass
class TrainHyperparameters:
    batch_size: int = 32
    window_size: int = 20
    stride: int = 1
    normalize_rul: bool = True
    sampler_seeds: tuple[int, int] = (42, 0)
    net_init_seed: int = 0
    train_rng_seed: int = 1
    drop_remainder: bool = False
    learning_rate: float = 1e-2
    n_epochs: int = 500
    beta_nll: float = 0.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    scheduler_alpha: float = 0.1


@dataclasses.dataclass(frozen=True)
class TestHyperparameters:
    """The `test_` prefix is used to distinguish these hyperparameters from the ones
    used for training."""

    test_rng_seed: int = 123
    test_n_soc0s: int = 10
    test_n_mc_samples: int = 100
    test_grid_crps_start: float = 0.0
    test_grid_crps_end: float = 5000.0
    test_grid_crps_num: int = 100
    test_simulation_dt: float = 20.0


@dataclasses.dataclass(frozen=True)
class DischargeData:
    time: np.ndarray
    soc: np.ndarray
    voltage: np.ndarray
    rul: np.ndarray


@dataclasses.dataclass(frozen=True)
class EvalTimeStamp:
    time: float
    target: np.ndarray
    samples_true: np.ndarray
    samples_pred: np.ndarray

    @staticmethod
    def _cdf(x: float, samples: np.ndarray) -> float:
        if len(samples) == 0:
            return 0.0

        sorted_samples = np.sort(samples)
        count = np.sum(sorted_samples <= x)
        cdf_value = count / len(sorted_samples)

        return cdf_value

    @property
    def average_pred(self) -> float:
        return np.mean(self.samples_pred)

    @property
    def ci_95_true(self) -> np.ndarray:
        """NOTE: could be private, but it is kept here for symmetry with the predicted CI."""
        return np.percentile(self.samples_true, [2.5, 97.5])

    @property
    def ci_95_pred(self) -> np.ndarray:
        return np.percentile(self.samples_pred, [2.5, 97.5])

    @property
    def squared_error(self) -> float:
        return (self.target - self.average_pred) ** 2

    @property
    def is_covered(self) -> bool:
        lower_bound_pred, upper_bound_pred = self.ci_95_pred
        return lower_bound_pred <= self.target <= upper_bound_pred

    def crps(self, x_grid: np.ndarray) -> float:
        F0 = np.array([self._cdf(x, self.samples_true) for x in x_grid])
        F1 = np.array([self._cdf(x, self.samples_pred) for x in x_grid])

        return np.trapezoid((F0 - F1) ** 2, x_grid)


@dataclasses.dataclass
class TestCaseResults:
    id: int
    eval_time_stamps: list[EvalTimeStamp]

    def __post_init__(self) -> None:
        self._times: np.ndarray = np.array([ets.time for ets in self.eval_time_stamps])

    @property
    def squared_errors(self) -> np.ndarray:
        return np.array([ets.squared_error for ets in self.eval_time_stamps[1:]])

    @property
    def coverage(self) -> float:
        return np.mean([ets.is_covered for ets in self.eval_time_stamps[1:]])

    @property
    def wsu(self) -> float:
        return np.dot(
            (
                np.array(
                    [
                        (ets_t.ci_95_pred[1] + ets_t1.ci_95_pred[1]) / 2
                        for (ets_t, ets_t1) in zip(
                            self.eval_time_stamps[2:], self.eval_time_stamps[1:-1]
                        )
                    ]
                )
                - np.array(
                    [
                        (ets_t.ci_95_pred[0] + ets_t1.ci_95_pred[0]) / 2
                        for (ets_t, ets_t1) in zip(
                            self.eval_time_stamps[2:], self.eval_time_stamps[1:-1]
                        )
                    ]
                )
            ),
            self._times[1:-1] - self._times[0],
        )

    @property
    def rmse(self) -> float:
        return np.sqrt(np.mean(self.squared_errors))

    def average_crps(self, x_grid: np.ndarray) -> float:
        return np.mean([ets.crps(x_grid=x_grid) for ets in self.eval_time_stamps])

    def plot_rul_over_time(self, ax: plt.Axes) -> None:
        ax.plot(
            self._times,
            [rt for rt in [ets.target for ets in self.eval_time_stamps]],
            label="True RUL",
        )
        ax.fill_between(
            self._times,
            [ets.ci_95_true[0] for ets in self.eval_time_stamps],
            [ets.ci_95_true[1] for ets in self.eval_time_stamps],
            alpha=0.3,
            label="True RUL CI",
        )
        ax.plot(
            self._times[1:],
            [ets.average_pred for ets in self.eval_time_stamps[1:]],
            "-o",
            label="Predicted RUL",
        )
        ax.fill_between(
            self._times[1:],
            [ets.ci_95_pred[0] for ets in self.eval_time_stamps[1:]],
            [ets.ci_95_pred[1] for ets in self.eval_time_stamps[1:]],
            alpha=0.3,
            label="Predicted RUL CI",
        )
        ax.set_title(f"Test Case {self.id}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RUL (s)")
        ax.set_ylim(bottom=0)
        ax.grid()
        ax.legend()

        return


def get_dataframes(
    train_path: pathlib.Path, test_path: pathlib.Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    # Split the train dataframe: if the run ID is < 100, then it goes in the training set, otherwise in the validation set. This way we ensure that the same RNG seed will always produce the same split.
    train_df, val_df = (
        train_df[train_df["run_id"] < 100],
        train_df[train_df["run_id"] >= 100],
    )
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def create_dataloaders(
    ds_train: DataSource,
    ds_val: DataSource,
    batch_size: int,
    sampler_seeds: tuple[int, int],
    drop_remainder: bool = False,
) -> tuple[Any, Any]:
    """Create Grain DataLoaders for training and validation."""

    sampler_train = grain.samplers.IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=True, seed=sampler_seeds[0]
    )
    dataloader_train = grain.DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[
            grain.transforms.Batch(batch_size=batch_size, drop_remainder=drop_remainder)
        ],
        worker_count=0,
    )

    sampler_val = grain.samplers.IndexSampler(
        num_records=len(ds_val), num_epochs=1, shuffle=False, seed=sampler_seeds[1]
    )
    dataloader_val = grain.DataLoader(
        data_source=ds_val,
        sampler=sampler_val,
        operations=[
            grain.transforms.Batch(batch_size=batch_size, drop_remainder=drop_remainder)
        ],
        worker_count=0,
    )

    return dataloader_train, dataloader_val


def get_test_case_data(test_path: pathlib.Path, test_case_id: int) -> DischargeData:
    """Return the discharge data for a given test case ID from the test CSV file.

    Args:
        test_path (pathlib.Path): Path to the test CSV file.
        test_case_id (int): ID of the test case to retrieve.

    Returns:
        DischargeData: Discharge data for the specified test case.
    """
    df_test = pd.read_csv(test_path)
    df_test_case_i = df_test[df_test["run_id"] == test_case_id]
    time = df_test_case_i["time"].values
    return DischargeData(
        time=time,
        soc=df_test_case_i["soc"].values,
        voltage=df_test_case_i["voltage"].values,
        rul=time[-1] - time,
    )


def run_discharges_from_intermediate_socs(
    soc_0s: np.ndarray, process_noise_std: float, dt: float
) -> Iterator[sb.SimulationResult]:
    for soc_0 in soc_0s:
        # TODO: simulation config parameters should be loaded from mlflow.
        config = sb.SimulationConfig(
            process_noise_distribution=lambda: np.random.normal(
                loc=0.0, scale=process_noise_std
            ),
            measurement_noise_distribution=lambda: 0.0,
            dt=dt,
            soc_0=soc_0,
        )
        result = sb.simulate_constant_capacity_simple(n_sim=100, config=config)
        yield result


def bar_plot_metrics_per_test_case(
    axes: Iterable[plt.Axes],
    test_case_results: list[TestCaseResults],
    rul_grid_crps: np.ndarray,
) -> None:
    """`axes` is expected to contain 4 subplot axes."""
    test_case_ids = [tcr.id for tcr in test_case_results]
    rmses = [tcr.rmse for tcr in test_case_results]
    coverages = [tcr.coverage for tcr in test_case_results]
    wsus = [tcr.wsu for tcr in test_case_results]
    average_crpss = [
        tcr.average_crps(x_grid=rul_grid_crps) for tcr in test_case_results
    ]

    metrics = {
        "RMSE": rmses,
        "Coverage": coverages,
        "WSU": wsus,
        "CRPS": average_crpss,
    }

    x = np.arange(len(test_case_ids))
    for ax, metric in zip(axes, metrics.keys()):
        ax.bar(x, metrics[metric])
        ax.set_xticks(x)
        ax.set_xticklabels(test_case_ids, rotation=45)
        ax.set_xlabel("Test Case ID")
        ax.set_title(metric)
        ax.grid()

    return
