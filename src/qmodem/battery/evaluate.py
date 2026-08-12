from __future__ import annotations

import pathlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from flax import nnx

from qmodem.battery.data_generation import TestCaseRULSamples, TestRULSamples
from qmodem.battery.scoring import (
    EvalTimeStamp,
    TestCaseResults,
    bar_plot_metrics_per_test_case,
)
from qmodem.data import DataScaler


class MCSampler(Protocol):
    """Structural interface for models that can produce Monte Carlo predictions.

    Every battery ``Net`` implements ``mc_sample`` with this signature; the concrete
    sampling logic (MC dropout, Bayesian weights, output Gaussian, ...) is model-specific.
    """

    def mc_sample(self, key: jax.Array, X: jax.Array, n_samples: int) -> jax.Array:
        """Return ``(n_samples, 1)`` predictions (in scaled space) for one window
        ``X``."""
        ...

    def train(self) -> None: ...

    def eval(self) -> None: ...


@dataclass(frozen=True)
class Hyperparameters:
    """The `test_` prefix is used to distinguish these hyperparameters from the ones
    used for training.

    Attributes:
        test_rng_seed: Seed for the PRNG key used to sample the model.
        test_n_soc0s: Number of intermediate starting SoCs to evaluate per test case.
        test_n_mc_samples: Number of Monte Carlo samples to draw per intermediate SoC.
        test_grid_crps_start: Start of the RUL grid for CRPS computation. Before this value,
            it is assumed that no particle reached the end of life.
        test_grid_crps_end: End of the RUL grid for CRPS computation. After this value,
            it is assumed that all particles reached the end of life.
        test_grid_crps_num: Number of points in the RUL grid for CRPS computation.
    """

    test_rng_seed: int = 123
    test_n_soc0s: int = 20
    test_n_mc_samples_simulator: int = 100
    test_n_mc_samples_model: int = 200
    test_grid_crps_start: float = 2_000.0
    test_grid_crps_end: float = 10_000.0
    # TODO: Specify the step, rather than the number of CRPS grid points.
    test_grid_crps_num: int = 100


def get_test_case_data(df_test: pd.DataFrame, test_case_id: int) -> pd.DataFrame:
    """Return the discharge data for a given test case ID from the test CSV file.

    Args:
        df_test (pd.DataFrame): DataFrame containing the test data.
        test_case_id (int): ID of the test case to retrieve.

    Returns:
        pd.DataFrame: Discharge data for the specified test case.
    """
    return df_test[df_test["run_id"] == test_case_id]


def restore_model_state(model: nnx.Module, train_run_id: str) -> None:
    """Restore the parameters of ``model`` in place from the ``best_model_state``
    checkpoint logged to the given MLflow training run."""
    abstract_state = nnx.state(model, nnx.Param)
    with tempfile.TemporaryDirectory() as tmp:
        artifact_dir = mlflow.artifacts.download_artifacts(
            run_id=train_run_id,
            artifact_path="best_model_state",
            dst_path=tmp,
        )
        checkpointer = ocp.StandardCheckpointer()
        restored_state = checkpointer.restore(
            pathlib.Path(artifact_dir), target=abstract_state
        )
    nnx.update(model, restored_state)


def evaluate_test_case(
    model: MCSampler,
    test_case_rul_samples: TestCaseRULSamples,
    test_case_id: int,
    test_data: pd.DataFrame,
    features: Sequence[str],
    x_scaler: DataScaler,
    y_scaler: DataScaler,
    hp: Hyperparameters,
    window_size: int,
    key: jax.Array,
) -> tuple[TestCaseResults, jax.Array]:
    """Evaluate a single test case over ``hp.test_n_soc0s`` intermediate starting SoCs.

    Args:
        model: The model implementing :class:`MCSampler`.
        test_case_rul_samples: RUL samples of the test case.
        test_case_id: Identifier of the test case (used to label the result).
        test_data: Discharge data of the test case.
        x_scaler: Fitted scaler for the input features.
        y_scaler: Fitted scaler for the target RUL values.
        config: Base simulation config (loaded from the data-generation run).
        hp: Evaluation hyperparameters.
        window_size: Voltage window length used by the model (from the training run).
        key: PRNG key; advanced and returned so it can be threaded across test cases.

    Returns:
        The per-test-case results and the advanced PRNG key.
    """
    # Order the test data by time to ensure correct evaluation.
    test_data = test_data.sort_values("time").reset_index(drop=True)

    soc0_idxs = np.linspace(
        0, len(test_data.time) - 1, num=hp.test_n_soc0s, dtype=np.int32
    )

    eval_time_stamps = []

    # First timestamp is treated separately, since there is no prediction for it.
    eval_time_stamps.append(
        EvalTimeStamp(
            time=test_data.time[soc0_idxs[0]],
            target=test_data.rul[soc0_idxs[0]],
            samples_true=test_case_rul_samples[
                0, :
            ],  # RUL samples for the first timestamp
            samples_pred=np.array([]),  # No prediction for the first timestamp
        )
    )

    for i in range(1, len(soc0_idxs)):
        if soc0_idxs[i] < window_size:
            time_window_start_idx = 0
        else:
            time_window_start_idx = soc0_idxs[i] - window_size
        previous_window = test_data[
            features
        ][
            time_window_start_idx : soc0_idxs[
                i
            ]  # time-window ends one step before the RUL timestamp
        ]
        previous_window = x_scaler.transform(previous_window)
        X = jnp.array(
            previous_window.reshape(1, -1, len(features))
        )  # shape (1, window_size, n_features)

        key, subkey = jax.random.split(key)
        samples_pred = np.array(model.mc_sample(subkey, X, hp.test_n_mc_samples_model))

        eval_time_stamps.append(
            EvalTimeStamp(
                time=test_data.time[soc0_idxs[i]],
                target=test_data.rul[soc0_idxs[i]],
                samples_true=test_case_rul_samples[
                    i, :
                ],  # RUL samples for the current timestamp
                samples_pred=y_scaler.inverse_transform(samples_pred),
            )
        )

    return (
        TestCaseResults(id=test_case_id, eval_time_stamps=eval_time_stamps),
        key,
    )


def log_evaluation_metrics(
    test_case_results: list[TestCaseResults], hp: Hyperparameters
) -> None:
    """Log the standard evaluation metrics and figures to the active MLflow run."""
    # Metric 1: plot RUL predictions with CI over time.
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    for test_case_result, ax in zip(test_case_results, axs):
        test_case_result.plot_rul_over_time(ax, legend=False)

    # Add a single legend for all subplots
    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])  # Make room for the figure legend

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
        np.mean([tcr.average_crps(x_grid=rul_grid_crps) for tcr in test_case_results]),
    )

    # Metric 6: bar plot of all metrics per test case.
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()
    bar_plot_metrics_per_test_case(
        axes=axs, test_case_results=test_case_results, rul_grid_crps=rul_grid_crps
    )
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_file="metrics_per_test_case.png")


def run_evaluation(
    *,
    model: MCSampler,
    test_data: pd.DataFrame,
    test_rul_samples: TestRULSamples,
    x_scaler: DataScaler,
    y_scaler: DataScaler,
    window_size: int,
    hp: Hyperparameters,
    features: Sequence[str] = ["voltage"],
) -> list[TestCaseResults]:
    """Run the full test-time evaluation and log results to MLflow.

    Args:
        model: Model implementing :class:`MCSampler`, already constructed by the caller.
        hp: Evaluation hyperparameters.
        mlflow_setup: MLflow setup for the evaluation run.
        raw_data_dir: Directory containing ``test.csv``.
        data_gen_run_id: MLflow run ID of the data-generation run holding the pickled
            simulation config.
        log_stream: In-memory log stream logged as an artifact at the end.
        train_mode: If True, put the model in train mode (e.g. to enable MC dropout);
            otherwise eval mode.
        features: List of feature names to use as input to the model.
        n_test_cases: Number of test cases to evaluate.

    Returns:
        List of per-test-case results.
    """
    # Random PRNG key for sampling the model.
    key = jax.random.key(hp.test_rng_seed)

    n_test_cases = test_data.run_id.nunique()
    test_case_results = []

    for test_case_id in range(n_test_cases):
        test_case_data = get_test_case_data(test_data, test_case_id=test_case_id)
        test_case_result, key = evaluate_test_case(
            model,
            test_case_rul_samples=test_rul_samples[f"test_case_{test_case_id}"],
            test_case_id=test_case_id,
            test_data=test_case_data,
            features=features,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            hp=hp,
            window_size=window_size,
            key=key,
        )
        test_case_results.append(test_case_result)

    return test_case_results
