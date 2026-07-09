from __future__ import annotations

import dataclasses
import io
import logging
import pathlib
import tempfile

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import orbax.checkpoint as ocp
import sklearn.preprocessing as skpp
from matplotlib import pyplot as plt

from qmodem.module import mc_sample
from qmodem.tracking import MLFlowSetup, track_mlflow
from scripts.battery.bnn_model import Net as BnnNet
from scripts.battery.commons import (
    DATA_GEN_RUN_ID,
    EvalTimeStamp,
    get_test_case_data,
    run_discharges_from_intermediate_socs,
)
from scripts.battery.hnn_model import Net as HnnNet
from scripts.battery.mcd_model import Net as McdNet
from scripts.battery.qavi_model import Net as QaviNet
from scripts.battery.qavi_model import WeightGenerator


@dataclasses.dataclass
class CompareCRPSHyperparameters:
    rng_seed: int = 0
    n_soc0s: int = 10
    n_mc_samples: int = 100
    grid_crps_start: float = 0.0
    grid_crps_end: float = 5000.0
    grid_crps_num: int = 100
    simulation_dt: float = 20.0


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

    TRAIN_RUN_IDS = {
        "hnn": "f58082c7c1fb413b8f7f90febac6ad64",
        "mcd": "29b4a225a607499eb7557f19b3c70b30",
        "bnn": "303ed092b22d498c9669a9099dfd761b",
        "qavi": "d7d63c2dccb54494bacbf2c0dbb078aa",
    }

    crps_collections = {
        "hnn": [],
        "mcd": [],
        "bnn": [],
        "qavi": [],
    }

    hp = CompareCRPSHyperparameters()

    rul_grid_crps = np.linspace(hp.grid_crps_start, hp.grid_crps_end, hp.grid_crps_num)

    for method, train_run_id in TRAIN_RUN_IDS.items():
        logging.info(f"Running tests for method: {method}")

        # Load the trianing run
        train_run = mlflow.get_run(train_run_id)

        # Load the mlflow run parameters
        run_params_training = train_run.data.params

        # Load the scaler fitted on the training data.
        scaler: skpp.MinMaxScaler = mlflow.sklearn.load_model(
            f"runs:/{train_run_id}/sklearn_scaler"
        )

        # Load the model
        if method == "bnn":
            model = BnnNet(
                rngs=nnx.Rngs(0),
                layer_type=run_params_training["conv_layer_type"],
                act_fn=getattr(nnx, run_params_training["activation_function"]),
            )
        elif method == "hnn":
            model = HnnNet(
                rngs=nnx.Rngs(0),
                act_fn=getattr(nnx, run_params_training["activation_function"]),
            )
        elif method == "mcd":
            model = McdNet(
                rngs=nnx.Rngs(0),
                act_fn=getattr(nnx, run_params_training["activation_function"]),
            )
        elif method == "qavi":
            w_gen = WeightGenerator(
                n_qubits=int(run_params_training["pqc_n_qubits"]),
                n_layers=int(run_params_training["pqc_n_layers"]),
                kernel_size=int(run_params_training["conv_kernel_size"]),
                in_features=1,
                out_features=int(run_params_training["conv_n_filters"]),
            )

            model = QaviNet(
                n_filters=int(run_params_training["conv_n_filters"]),
                kernel_size=int(run_params_training["conv_kernel_size"]),
                generator=w_gen,
                act_fn=getattr(nnx, run_params_training["activation_function"]),
                rngs=nnx.Rngs(0),
            )  # RNGs won't be used for inference, so the seed is arbitrary.

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

        # Random PRNG key for sampling the model.
        key = jax.random.key(hp.rng_seed)

        model.train()  # Enables MC Dropout.
        for test_case_id in range(10):
            test_data = get_test_case_data(
                RAW_DATA_DIR / "test.csv", test_case_id=test_case_id
            )

            soc0_idxs = np.linspace(
                0, len(test_data.time) - 1, num=hp.n_soc0s, dtype=np.int32
            )

            # Load process noise parameters from the data generation run.
            data_gen_run = mlflow.get_run(DATA_GEN_RUN_ID)
            sims_iterator = run_discharges_from_intermediate_socs(
                soc_0s=test_data.soc[soc0_idxs],
                process_noise_std=float(data_gen_run.data.params["process_noise_std"]),
                dt=hp.simulation_dt,
            )

            # Discard the first sample, since there is no prediction for it.
            next(sims_iterator)

            i = 1
            for sr in sims_iterator:
                previous_voltage_window = test_data.voltage[
                    soc0_idxs[i] - int(run_params_training["window_size"]) : soc0_idxs[
                        i
                    ]
                    + 1
                ]

                X = jnp.array(previous_voltage_window.reshape(1, -1, 1))

                # Keys for sampling the model weights and the output Gaussian distribution.
                splits = jax.random.split(key, num=2 * hp.n_mc_samples + 1)
                key, keys_weights, keys_noise = (
                    splits[0],
                    splits[1 : hp.n_mc_samples + 1],
                    splits[hp.n_mc_samples + 1 :],
                )

                samples_pred = mc_sample(
                    model, X, key_weights=keys_weights, key_noise=keys_noise
                )

                crps_collections[method].append(
                    EvalTimeStamp(
                        time=test_data.time[soc0_idxs[i]],
                        target=test_data.rul[soc0_idxs[i]],
                        samples_true=sr.times_eod - sr.times[0],
                        samples_pred=scaler.inverse_transform(
                            samples_pred
                        ),  # Placeholder, will be filled later
                    ).crps(rul_grid_crps)
                )
                i += 1

    # Box plot of the CRPS for the different methods
    fig, ax = plt.subplots()
    ax.boxplot(
        x=[crps_collections[m] for m in crps_collections.keys()],
        tick_labels=crps_collections.keys(),
    )
    ax.set_ylabel("CRPS [s]")
    ax.set_title("CRPS Statistics")
    ax.grid()

    mlflow_setup = MLFlowSetup(
        experiment_name="checkpoint_phme_2026",
        run_name="compare_crps",
        run_description="Produce box plots with the statistics across time and test cases for each method.",
        tags={
            "case_study": "battery",
            "stage": "publication",
        },
    )

    with track_mlflow(setup=mlflow_setup):
        mlflow.log_params(dataclasses.asdict(hp))

        # Log the CRPS box plot image
        mlflow.log_figure(fig, artifact_file="crps_box.png")

        mlflow.log_text(log_stream.getvalue(), artifact_file="test_log.txt")


if __name__ == "__main__":
    main()
