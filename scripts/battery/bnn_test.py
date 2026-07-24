from __future__ import annotations

import io
import logging
import os
import pathlib

import flax.nnx as nnx
import mlflow
from dotenv import load_dotenv

from qmodem.battery.evaluate import Hyperparameters, run_evaluation
from qmodem.battery.models import BayesianCNN
from qmodem.tracking import MLFlowSetup


def main() -> None:
    load_dotenv()

    # TODO: move to a common function for all scripts
    log_stream = io.StringIO()
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=[
            logging.StreamHandler(),  # console (stderr)
            logging.StreamHandler(log_stream),  # in-memory stream for MLflow logging
        ],
    )

    TRAIN_RUN_ID = "303ed092b22d498c9669a9099dfd761b"

    hp = Hyperparameters()

    mlflow_setup = MLFlowSetup(
        experiment_name="checkpoint_phme_2026", run_id=TRAIN_RUN_ID
    )

    # Build the model from the training-run parameters before opening the tracking
    # context. Setting the tracking URI is required for `get_run` to resolve the run.
    mlflow.set_tracking_uri(mlflow_setup.backend_store)
    params = mlflow.get_run(TRAIN_RUN_ID).data.params
    model = BayesianCNN(
        rngs=nnx.Rngs(0),
        layer_type=params["conv_layer_type"],
        act_fn=getattr(nnx, params["activation_function"]),
    )  # RNGs won't be used for inference, so the seed is arbitrary.

    run_evaluation(
        model=model,
        mlflow_setup=mlflow_setup,
        hp=hp,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID"],
        log_stream=log_stream,
        train_mode=False,  # Enables MC Dropout / weight sampling.
    )


if __name__ == "__main__":
    main()
