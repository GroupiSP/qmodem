from __future__ import annotations

import dataclasses
import pathlib
import tempfile
from typing import Iterator

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import sklearn.preprocessing as skpp

from qmodem.tracking import MLFlowSetup, track_mlflow

from .hnn_model import Net


@dataclasses.dataclass(frozen=True)
class EvalTimeStamp:
    time: float
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
    def average_true(self) -> float:
        return np.mean(self.samples_true)

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
        return (self.average_true - self.average_pred) ** 2

    @property
    def is_covered(self) -> bool:
        lower_bound_pred, upper_bound_pred = self.ci_95_pred
        return lower_bound_pred <= self.average_true <= upper_bound_pred

    def crps(self, x_grid: np.ndarray) -> float:
        F0 = np.array([self._cdf(x, self.samples_true) for x in x_grid])
        F1 = np.array([self._cdf(x, self.samples_pred) for x in x_grid])

        return np.trapz((F0 - F1) ** 2, x_grid)


@dataclasses.dataclass(frozen=True)
class TestCaseResults:
    eval_time_stamps: list[EvalTimeStamp]

    def __post_init__(self) -> None:
        self._times: np.ndarray = np.array([ets.time for ets in self.eval_time_stamps])

    @property
    def squared_errors(self) -> np.ndarray:
        return np.array([ets.squared_error for ets in self.eval_time_stamps])

    @property
    def coverage(self) -> float:
        return np.mean([ets.is_covered for ets in self.eval_time_stamps])

    @property
    def wsu(self) -> float:
        return np.dot(
            (
                (
                    self.eval_time_stamps[1:].ci_95_pred[1]
                    + self.eval_time_stamps[:-1].ci_95_pred[1]
                )
                / 2
                - (
                    self.eval_time_stamps[1:].ci_95_pred[0]
                    + self.eval_time_stamps[:-1].ci_95_pred[0]
                )
                / 2
            ),
            self._times[1:] - self._times[0],
        )

    @property
    def rmse(self) -> float:
        return np.sqrt(np.mean(self.squared_errors))

    def average_crps(self, x_grid: np.ndarray) -> float:
        return np.mean([ets.crps(x_grid=x_grid) for ets in self.eval_time_stamps])


@dataclasses.dataclass(frozen=True)
class DischargeData:
    time: np.ndarray
    soc: np.ndarray
    voltage: np.ndarray
    rul: np.ndarray


@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    """The `test_` prefix is used to distinguish these hyperparameters from the ones
    used for training."""

    test_rng_seed: int = 123
    test_n_soc0s: int = 10
    test_n_mc_samples: int = 100
    test_grid_crps_start: float = 0.0
    test_grid_crps_end: float = 5000.0
    test_grid_crps_num: int = 100


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
    soc_0s: np.ndarray,
) -> Iterator[les.SimulationResult]:
    for soc_0 in soc_0s:
        # TODO: simulation config parameters should be loaded from mlflow.
        config = les.SimulationConfig(
            process_noise_distribution=lambda: np.random.normal(loc=0.0, scale=3e-3),
            measurement_noise_distribution=lambda: 0.0,
            dt=20.0,
            soc_0=soc_0,
        )
        result = les.simulate_constant_capacity_simple(n_sim=100, config=config)
        yield result


def mc_sample_model(
    model: Net, X: np.ndarray, n_samples: int, rng_key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    mu, var = model(X, rngs=nnx.Rngs(dropout=rng_key)).squeeze()  # Shape (2,)

    rng_key, _ = jax.random.split(rng_key)
    samples = mu + jnp.sqrt(var) * jax.random.normal(rng_key, shape=(n_samples, 1))
    return samples, rng_key


def main() -> None:
    RAW_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    RUN_ID = "b6e03cd77f9f4ab48425225b39e154dc"

    hp = Hyperparameters()

    with track_mlflow(
        MLFlowSetup(experiment_name="battery_default", run_id=RUN_ID)
    ) as run:
        # Load the mlflow run parameters
        run_params_training = run.data.params

        # Load the scaler fitted on the training data.
        scaler: skpp.MinMaxScaler = mlflow.sklearn.load_model(
            f"runs:/{RUN_ID}/sklearn_scaler"
        )

        # Load the model
        model = Net(rngs=nnx.Rngs(0))
        abstract_state = nnx.state(model, nnx.Param)

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = mlflow.artifacts.download_artifacts(
                run_id=RUN_ID,
                artifact_path="best_model_state",
                dst_path=tmp,
            )
            checkpointer = ocp.StandardCheckpointer()
            restored_state = checkpointer.restore(
                pathlib.Path(artifact_dir), target=abstract_state
            )

        nnx.update(model, restored_state)

        # Create a subplot figure to compare the predictions over the different test cases.
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        # Random PRNG key for sampling the model.
        key = jax.random.PRNGKey(hp.test_rng_seed)

        # rul_grid_crps = np.linspace(
        #     hp.test_grid_crps_start, hp.test_grid_crps_end, hp.test_grid_crps_num
        # )

        for test_case_id in range(10):
            test_data = get_test_case_data(
                RAW_DATA_DIR / "test.csv", test_case_id=test_case_id
            )

            N_t = len(test_data.time)
            soc0_idxs = np.linspace(0, N_t - 1, num=hp.test_n_soc0s, dtype=np.int32)

            # True RUL, distribution and bounds.
            ruls_true = test_data.rul[soc0_idxs]
            ruls_lower_true = []
            ruls_upper_true = []
            for sr in run_discharges_from_intermediate_socs(
                soc_0s=test_data.soc[soc0_idxs],
            ):
                samples_true = sr.times_eod - sr.times[0]
                ruls_lower_true.append(np.percentile(samples_true, 2.5))
                ruls_upper_true.append(np.percentile(samples_true, 97.5))

            # Predicted RUL, distribution and bounds.
            ruls_pred = []
            ruls_lower_pred = []
            ruls_upper_pred = []
            for int_idx in soc0_idxs[1:]:
                # Get the voltage window that is immediately prior to the test timestamp
                previous_voltage_window = test_data.voltage[
                    int_idx - int(run_params_training["window_size"]) : int_idx + 1
                ]
                X = jnp.array(previous_voltage_window.reshape(1, -1, 1))
                samples_pred, key = mc_sample_model(
                    model, X, n_samples=hp.test_n_mc_samples, rng_key=key
                )
                samples_pred = scaler.inverse_transform(samples_pred)

                y = jnp.average(samples_pred, axis=0, keepdims=True)
                ruls_pred.append(y.item())

                ruls_lower_pred.append(jnp.percentile(samples_pred, 2.5))
                ruls_upper_pred.append(jnp.percentile(samples_pred, 97.5))

            # Plot true and predicted RULs against time for the current test case.
            # TODO: [refac] move plotting to a separate function
            axes[test_case_id].plot(
                test_data.time[soc0_idxs], ruls_true, label="True RUL"
            )
            axes[test_case_id].fill_between(
                test_data.time[soc0_idxs],
                ruls_lower_true,
                ruls_upper_true,
                alpha=0.3,
                label="True RUL CI",
            )
            axes[test_case_id].plot(
                test_data.time[soc0_idxs[1:]], ruls_pred, "-o", label="Predicted RUL"
            )
            axes[test_case_id].fill_between(
                test_data.time[soc0_idxs[1:]],
                ruls_lower_pred,
                ruls_upper_pred,
                alpha=0.3,
                label="Predicted RUL CI",
            )
            axes[test_case_id].set_title(f"Test Case {test_case_id}")
            axes[test_case_id].set_xlabel("Time (s)")
            axes[test_case_id].set_ylabel("RUL (s)")
            axes[test_case_id].set_ylim(bottom=0)
            axes[test_case_id].grid()
            axes[test_case_id].legend()

        fig.tight_layout()
        mlflow.log_figure(fig, artifact_file="rul_predictions_over_test_cases.png")

        # TODO: [feat] Produce the remaining metrics per test case, average them and log them to mlflow.


if __name__ == "__main__":
    main()
