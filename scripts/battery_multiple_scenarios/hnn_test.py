from __future__ import annotations

import os
import pathlib

import flax.nnx as nnx
from dotenv import load_dotenv

from qmodem.battery.evaluate import Hyperparameters, run_evaluation
from qmodem.battery.models import HeteroscedasticCNN
from qmodem.tracking import MLFlowSetup
from qmodem.utils import setup_script_logging


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()

    TRAIN_RUN_ID = "6483d8bfd2fe4cd8a0d7812216031bc8"

    hp = Hyperparameters()

    mlflow_setup = MLFlowSetup(
        experiment_name="refactoring_jul_2026", run_id=TRAIN_RUN_ID
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
