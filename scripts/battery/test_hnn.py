from __future__ import annotations

import dataclasses
import pathlib
import tempfile
from typing import Iterable, Iterator

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
    test_process_noise_std: float = 3e-3
    test_simulation_dt: float = 20.0


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
) -> Iterator[les.SimulationResult]:
    for soc_0 in soc_0s:
        # TODO: simulation config parameters should be loaded from mlflow.
        config = les.SimulationConfig(
            process_noise_distribution=lambda: np.random.normal(
                loc=0.0, scale=process_noise_std
            ),
            measurement_noise_distribution=lambda: 0.0,
            dt=dt,
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
        # TODO: [refac] Enclose model loading into a function and move it to commons.
        model = Net(
            rngs=nnx.Rngs(0)
        )  # RNGs won't be used for inference, so the seed is arbitrary.
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

        # Random PRNG key for sampling the model.
        key = jax.random.PRNGKey(hp.test_rng_seed)

        test_case_results = []
        for test_case_id in range(10):
            test_data = get_test_case_data(
                RAW_DATA_DIR / "test.csv", test_case_id=test_case_id
            )

            soc0_idxs = np.linspace(
                0, len(test_data.time) - 1, num=hp.test_n_soc0s, dtype=np.int32
            )

            # True RUL, distribution and bounds.
            eval_time_stamps = []

            sims_iterator = run_discharges_from_intermediate_socs(
                soc_0s=test_data.soc[soc0_idxs],
                process_noise_std=hp.test_process_noise_std,
                dt=hp.test_simulation_dt,
            )

            # First timestamp is treated separately, since there is no prediction for it.
            sr_0 = next(sims_iterator)
            eval_time_stamps.append(
                EvalTimeStamp(
                    time=test_data.time[soc0_idxs[0]],
                    target=test_data.rul[soc0_idxs[0]],
                    samples_true=sr_0.times_eod - sr_0.times[0],
                    samples_pred=np.array([]),  # No prediction for the first timestamp
                )
            )

            i = 1
            for sr in sims_iterator:
                previous_voltage_window = test_data.voltage[
                    soc0_idxs[i] - int(run_params_training["window_size"]) : soc0_idxs[
                        i
                    ]
                    + 1
                ]

                X = jnp.array(previous_voltage_window.reshape(1, -1, 1))

                key, _ = jax.random.split(key)
                samples_pred, key = mc_sample_model(
                    model, X, n_samples=hp.test_n_mc_samples, rng_key=key
                )

                eval_time_stamps.append(
                    EvalTimeStamp(
                        time=test_data.time[soc0_idxs[i]],
                        target=test_data.rul[soc0_idxs[i]],
                        samples_true=sr.times_eod - sr.times[0],
                        samples_pred=scaler.inverse_transform(
                            samples_pred
                        ),  # Placeholder, will be filled later
                    )
                )
                i += 1

            test_case_results.append(
                TestCaseResults(id=test_case_id, eval_time_stamps=eval_time_stamps)
            )

        # Log parameters with MLFlow.
        mlflow.log_params(dataclasses.asdict(hp))

        # Metric 1: plot RUL predictions with CI over time.
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()

        for test_case_result, ax in zip(test_case_results, axes):
            test_case_result.plot_rul_over_time(ax)

        fig.tight_layout()
        mlflow.log_figure(fig, artifact_file="rul_predictions_over_test_cases.png")

        # Metric 2: average RMSE
        mlflow.log_metric(
            "rmse_average",
            np.mean([tcr.rmse for tcr in test_case_results]),
        )

        # Metric 3: average coverage
        mlflow.log_metric(
            "coverage_average",
            np.mean([tcr.coverage for tcr in test_case_results]),
        )

        # Metric 4: average WSU
        mlflow.log_metric(
            "wsu_average",
            np.mean([tcr.wsu for tcr in test_case_results]),
        )

        # Metric 5: average CRPS over a common grid.
        rul_grid_crps = np.linspace(
            hp.test_grid_crps_start, hp.test_grid_crps_end, hp.test_grid_crps_num
        )
        mlflow.log_metric(
            "crps_average",
            np.mean(
                [tcr.average_crps(x_grid=rul_grid_crps) for tcr in test_case_results]
            ),
        )

        # Metric 6: bar plot of all metrics per test case.
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.flatten()
        bar_plot_metrics_per_test_case(
            axes=axes, test_case_results=test_case_results, rul_grid_crps=rul_grid_crps
        )
        fig.tight_layout()
        mlflow.log_figure(fig, artifact_file="metrics_per_test_case.png")


if __name__ == "__main__":
    main()
