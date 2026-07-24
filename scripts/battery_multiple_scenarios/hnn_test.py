from __future__ import annotations

import io
import logging
import os
import pathlib

import flax.nnx as nnx
from dotenv import load_dotenv

from qmodem.battery.evaluate import Hyperparameters, run_evaluation
from qmodem.battery.models import HeteroscedasticCNN
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

    TRAIN_RUN_ID = "d6d142895f27463292ebe8023dbc8e06"

    hp = Hyperparameters(test_n_soc0s=20)

    mlflow_setup = MLFlowSetup(
        experiment_name="variable_loading_conditions", run_id=TRAIN_RUN_ID
    )

    model = HeteroscedasticCNN(
        rngs=nnx.Rngs(0)
    )  # RNGs won't be used for inference, so the seed is arbitrary.

    run_evaluation(
        model=model,
        mlflow_setup=mlflow_setup,
        hp=hp,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
        log_stream=log_stream,
        train_mode=False,  # Deterministic network: aleatoric uncertainty only.
    )


if __name__ == "__main__":
    main()
