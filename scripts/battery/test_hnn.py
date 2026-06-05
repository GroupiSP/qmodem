from __future__ import annotations

import dataclasses
import pathlib
import tempfile
from typing import Iterator

import flax.nnx as nnx
import jax
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
class DischargeData:
    time: np.ndarray
    soc: np.ndarray
    voltage: np.ndarray
    rul: np.ndarray


@dataclasses.dataclass(frozen=True)
class TestCaseMetrics:
    squared_errors: np.ndarray
    coverage: float
    wsu: float
    crps: np.ndarray


@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    test_rng_seed: int = 123
    test_n_soc0s: int = 200


def compute_squared_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return (y_true - y_pred) ** 2


def compute_coverage(
    y_true: np.ndarray, lower_bounds_pred: np.ndarray, upper_bounds_pred: np.ndarray
) -> float:
    return np.mean((y_true >= lower_bounds_pred) & (y_true <= upper_bounds_pred))


def compute_wsu(
    times: np.ndarray, lower_bounds_pred: np.ndarray, upper_bounds_pred: np.ndarray
) -> float:
    return np.dot(
        (
            (upper_bounds_pred[1:] + upper_bounds_pred[:-1]) / 2
            - (lower_bounds_pred[1:] + lower_bounds_pred[:-1]) / 2
        ),
        times[1:] - times[0],
    )


def compute_crps_at_timestamp(
    samples_true: np.ndarray, samples_pred: np.ndarray, x_grid: np.ndarray
) -> float:
    def _cdf(x: float, samples: np.ndarray) -> float:
        if len(samples) == 0:
            return 0.0

        sorted_samples = np.sort(samples)
        count = np.sum(sorted_samples <= x)
        cdf_value = count / len(sorted_samples)

        return cdf_value

    F0 = np.array([_cdf(x, samples_true) for x in x_grid])
    F1 = np.array([_cdf(x, samples_pred) for x in x_grid])

    crps_value = np.trapz((F0 - F1) ** 2, x_grid)
    return crps_value


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


# def get_rul_confidence_interval(times_eod: np.ndarray) -> tuple[float, float]:
#     """Compute the 95% confidence interval for the RUL predictions for one stochastic
#     battery discharge simulation.

#     Args:
#         times_eod (np.ndarray): Array of shape (n_samples, n_time_steps) containing the predicted EOD times.
#     Returns:
#         tuple[float, float]: Lower and upper bounds of the 95% confidence interval for the RUL predictions.
#     """
#     lower_bound = np.percentile(times_eod, 2.5, axis=0)
#     upper_bound = np.percentile(times_eod, 97.5, axis=0)
#     return lower_bound, upper_bound


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

        for test_case_id in range(10):
            test_data = get_test_case_data(
                RAW_DATA_DIR / "test.csv", test_case_id=test_case_id
            )

            N_t = len(test_data.time)
            soc0_idxs = np.linspace(0, N_t - 1, num=hp.test_n_soc0s, dtype=np.int32)

            ruls_true = test_data.rul[soc0_idxs]
            ruls_pred = []
            for int_idx in soc0_idxs[1:]:
                # Get the voltage window that is immediately prior to the test timestamp
                previous_voltage_window = test_data.voltage[
                    int_idx - int(run_params_training["window_size"]) : int_idx + 1
                ]
                X = previous_voltage_window.reshape(1, -1, 1)

                key, _ = jax.random.split(key)
                y = model(X, rngs=nnx.Rngs(dropout=key))
                y = scaler.inverse_transform(y)

                ruls_pred.append(y[0, 0])

            # Plot true and predicted RULs against time for the current test case.
            axes[test_case_id].plot(
                test_data.time[soc0_idxs], ruls_true, label="True RUL"
            )
            axes[test_case_id].plot(
                test_data.time[soc0_idxs[1:]], ruls_pred, label="Predicted RUL"
            )

        mlflow.log_figure(fig, artifact_file="rul_predictions_over_test_cases.png")

        # Run multiple discharge simulations from intermediate SoCs to
        # reconstruct the true RUL distribution over time.
        # sim_results = [
        #     sr
        #     for sr in run_discharges_from_intermediate_socs(
        #         soc_0s=test_data.soc[soc0_idxs],
        #     )
        # ]


if __name__ == "__main__":
    main()
