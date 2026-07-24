from __future__ import annotations

import io
import logging
import os
import pathlib

import flax.nnx as nnx
import mlflow
from dotenv import load_dotenv

from qmodem.battery.evaluate import Hyperparameters, run_evaluation
from qmodem.battery.models import QuantumVICNN, WeightGenerator
from qmodem.tracking import MLFlowSetup


def main() -> None:
    load_dotenv()

    log_stream = io.StringIO()
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=[
            logging.StreamHandler(),  # console (stderr)
            logging.StreamHandler(log_stream),  # in-memory stream for MLflow logging
        ],
    )

    TRAIN_RUN_ID = "8f52cdeced0a4e8b965ffa4c183b8479"

    hp = Hyperparameters(test_n_soc0s=20)

    mlflow_setup = MLFlowSetup(
        experiment_name="variable_loading_conditions", run_id=TRAIN_RUN_ID
    )

    # Build the model from the training-run parameters before opening the tracking
    # context. Setting the tracking URI is required for `get_run` to resolve the run.
    mlflow.set_tracking_uri(mlflow_setup.backend_store)
    params = mlflow.get_run(TRAIN_RUN_ID).data.params

    w_gen = WeightGenerator(
        n_qubits=int(params["pqc_n_qubits"]),
        n_layers=int(params["pqc_n_layers"]),
        kernel_size=int(params["conv_kernel_size"]),
        in_features=1,
        out_features=int(params["conv_n_filters"]),
    )
    model = QuantumVICNN(
        n_filters=int(params["conv_n_filters"]),
        kernel_size=int(params["conv_kernel_size"]),
        generator=w_gen,
        act_fn=getattr(nnx, params["activation_function"]),
        rngs=nnx.Rngs(0),
    )  # RNGs won't be used for inference, so the seed is arbitrary.

    run_evaluation(
        model=model,
        mlflow_setup=mlflow_setup,
        hp=hp,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
        log_stream=log_stream,
        # TODO: can QAVI sample in eval mode?
        train_mode=True,  # Enables weight sampling.
    )


if __name__ == "__main__":
    main()
