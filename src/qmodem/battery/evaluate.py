from __future__ import annotations

import dataclasses
import io
import pathlib
import tempfile
from typing import Protocol

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import simbat as sb
import sklearn.preprocessing as skpp

from qmodem.battery.data_generation import (
    load_simulation_config,
    run_discharges_from_intermediate_socs,
)
from qmodem.battery.scoring import (
    DischargeData,
    EvalTimeStamp,
    TestCaseResults,
    bar_plot_metrics_per_test_case,
)
from qmodem.tracking import MLFlowSetup, track_mlflow


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


@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    """The `test_` prefix is used to distinguish these hyperparameters from the ones
    used for training."""

    test_rng_seed: int = 123
    test_n_soc0s: int = 20
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
    *,
    test_case_id: int,
    test_data: DischargeData,
    scaler: skpp.MinMaxScaler,
    config: sb.SimulationConfig,
    hp: Hyperparameters,
    window_size: int,
    key: jax.Array,
) -> tuple[TestCaseResults, jax.Array]:
    """Evaluate a single test case over ``hp.test_n_soc0s`` intermediate starting SoCs.

    Args:
        model: The model implementing :class:`MCSampler`.
        test_case_id: Identifier of the test case (used to label the result).
        test_data: Discharge data of the test case.
        scaler: Fitted scaler used to map predictions back to RUL units.
        config: Base simulation config (loaded from the data-generation run).
        hp: Evaluation hyperparameters.
        window_size: Voltage window length used by the model (from the training run).
        key: PRNG key; advanced and returned so it can be threaded across test cases.

    Returns:
        The per-test-case results and the advanced PRNG key.
    """
    soc0_idxs = np.linspace(
        0, len(test_data.time) - 1, num=hp.test_n_soc0s, dtype=np.int32
    )

    # Simulate the "true" future from each intermediate starting state. Both soc_0 and
    # t_0 are injected: t_0 is required for time-dependent policies (multi-scenario) and
    # harmless for constant policies since the RUL (times_eod - times[0]) is
    # offset-invariant. Measurement noise is kept identical to data generation.
    overrides = [
        {"soc_0": test_data.soc[idx], "t_0": test_data.time[idx]} for idx in soc0_idxs
    ]
    sims_iterator = run_discharges_from_intermediate_socs(config, overrides)

    eval_time_stamps = []

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
            soc0_idxs[i] - window_size : soc0_idxs[i] + 1
        ]
        X = jnp.array(previous_voltage_window.reshape(1, -1, 1))

        key, subkey = jax.random.split(key)
        samples_pred = model.mc_sample(subkey, X, hp.test_n_mc_samples)

        eval_time_stamps.append(
            EvalTimeStamp(
                time=test_data.time[soc0_idxs[i]],
                target=test_data.rul[soc0_idxs[i]],
                samples_true=sr.times_eod - sr.times[0],
                samples_pred=scaler.inverse_transform(samples_pred),
            )
        )
        i += 1

    return (
        TestCaseResults(id=test_case_id, eval_time_stamps=eval_time_stamps),
        key,
    )


def log_evaluation_metrics(
    test_case_results: list[TestCaseResults], hp: Hyperparameters
) -> None:
    """Log the standard evaluation metrics and figures to the active MLflow run."""
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
        np.mean([tcr.average_crps(x_grid=rul_grid_crps) for tcr in test_case_results]),
    )

    # Metric 6: bar plot of all metrics per test case.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes = axes.flatten()
    bar_plot_metrics_per_test_case(
        axes=axes, test_case_results=test_case_results, rul_grid_crps=rul_grid_crps
    )
    fig.tight_layout()
    mlflow.log_figure(fig, artifact_file="metrics_per_test_case.png")


def run_evaluation(
    *,
    model: MCSampler,
    mlflow_setup: MLFlowSetup,
    hp: Hyperparameters,
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    log_stream: io.StringIO,
    train_run_id: str | None = None,
    train_mode: bool = True,
    n_test_cases: int = 10,
) -> None:
    """Run the full test-time evaluation and log results to MLflow.

    Args:
        model: Model implementing :class:`MCSampler`, already constructed by the caller.
        mlflow_setup: MLflow run configuration. When ``train_run_id`` is None, its
            ``run_id`` is assumed to be the training run (results are written onto it).
        hp: Evaluation hyperparameters.
        raw_data_dir: Directory containing ``test.csv``.
        data_gen_run_id: MLflow run ID of the data-generation run holding the pickled
            simulation config.
        log_stream: In-memory log stream logged as an artifact at the end.
        train_run_id: Training run holding the scaler and model checkpoint. Defaults to
            ``mlflow_setup.run_id``.
        train_mode: If True, put the model in train mode (e.g. to enable MC dropout);
            otherwise eval mode.
        n_test_cases: Number of test cases to evaluate.
    """
    train_run_id = train_run_id or mlflow_setup.run_id
    if train_run_id is None:
        raise ValueError(
            "train_run_id must be provided, either explicitly or via mlflow_setup.run_id."
        )

    with track_mlflow(setup=mlflow_setup) as run:
        run_params_training = run.data.params

        # Load the scaler fitted on the training data.
        scaler: skpp.MinMaxScaler = mlflow.sklearn.load_model(
            f"runs:/{train_run_id}/sklearn_scaler"
        )

        restore_model_state(model, train_run_id)

        # Load the simulation config used to generate the test cases.
        config = load_simulation_config(data_gen_run_id)

        window_size = int(run_params_training["window_size"])

        # Random PRNG key for sampling the model.
        key = jax.random.key(hp.test_rng_seed)

        if train_mode:
            model.train()
        else:
            model.eval()

        test_case_results = []
        for test_case_id in range(n_test_cases):
            test_data = get_test_case_data(
                raw_data_dir / "test.csv", test_case_id=test_case_id
            )
            test_case_result, key = evaluate_test_case(
                model,
                test_case_id=test_case_id,
                test_data=test_data,
                scaler=scaler,
                config=config,
                hp=hp,
                window_size=window_size,
                key=key,
            )
            test_case_results.append(test_case_result)

        # Log parameters and metrics with MLflow.
        mlflow.log_params(dataclasses.asdict(hp))
        log_evaluation_metrics(test_case_results, hp)
        mlflow.log_text(log_stream.getvalue(), artifact_file="test_log.txt")
