from __future__ import annotations

import dataclasses
import io
import logging
import pathlib
import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import orbax.checkpoint as ocp
import sklearn.preprocessing as skpp

from qmodem.tracking import MLFlowSetup, track_mlflow
from scripts.battery.commons import (
    DATA_GEN_RUN_ID,
    EvalTimeStamp,
    TestCaseResults,
    TestHyperparameters,
    bar_plot_metrics_per_test_case,
    get_test_case_data,
    run_discharges_from_intermediate_socs,
)
from scripts.battery.qavi_model import Net


def mc_sample_model(
    model: Net, X: np.ndarray, n_samples: int, rng_key: jax.Array
) -> tuple[jax.Array, jax.Array]:
    samples = []
    for _ in range(n_samples):
        rng_key, _ = jax.random.split(rng_key)
        # TODO: rng_key does not need to be associated to a specific stream.
        # The same should be corrected for dropout.
        mu, var = model(X, rngs=nnx.Rngs(params=rng_key)).squeeze()  # Shape (2,)
        samples.append(mu + jnp.sqrt(var) * jax.random.normal(rng_key, shape=(1,)))
    samples = jnp.array(samples).reshape(-1, 1)  # Shape (n_samples, 1)

    return samples, rng_key


def main() -> None:
    log_stream = io.StringIO()
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=[
            logging.StreamHandler(),  # console (stderr)
            logging.StreamHandler(log_stream),  # in-memory stream for MLflow logging
        ],
    )

    RAW_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    TRAIN_RUN_ID = "c95ecc12b8b5468daa8523259ed1fe21"

    hp = TestHyperparameters()

    mlflow_setup = MLFlowSetup(experiment_name="variance_tracking", run_id=TRAIN_RUN_ID)

    with track_mlflow(setup=mlflow_setup) as run:
        # Load the mlflow run parameters
        run_params_training = run.data.params

        # Load the scaler fitted on the training data.
        scaler: skpp.MinMaxScaler = mlflow.sklearn.load_model(
            f"runs:/{TRAIN_RUN_ID}/sklearn_scaler"
        )

        # Load the model
        # TODO: [refac] Enclose model loading into a function and move it to commons.
        model = Net(
            rngs=nnx.Rngs(0)
        )  # RNGs won't be used for inference, so the seed is arbitrary.
        abstract_state = nnx.state(model, nnx.Param)

        with tempfile.TemporaryDirectory() as tmp:
            artifact_dir = mlflow.artifacts.download_artifacts(
                run_id=TRAIN_RUN_ID,
                artifact_path="best_model_state",
                dst_path=tmp,
            )
            checkpointer = ocp.StandardCheckpointer()
            restored_state = checkpointer.restore(
                pathlib.Path(artifact_dir), target=abstract_state
            )

        nnx.update(model, restored_state)

        # Random PRNG key for sampling the model.
        key = jax.random.key(hp.test_rng_seed)

        model.train()  # Enables MC Dropout.
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

            # Load process noise parameters from the data generation run.
            data_gen_run = mlflow.get_run(DATA_GEN_RUN_ID)
            sims_iterator = run_discharges_from_intermediate_socs(
                soc_0s=test_data.soc[soc0_idxs],
                process_noise_std=float(data_gen_run.data.params["process_noise_std"]),
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

        mlflow.log_text(log_stream.getvalue(), artifact_file="test_log.txt")


if __name__ == "__main__":
    main()
