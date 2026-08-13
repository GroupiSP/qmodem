from __future__ import annotations

import os
import pathlib

import flax.nnx as nnx
from dotenv import load_dotenv

from qmodem.battery.evaluate import TestHyperparameters, run_evaluation
from qmodem.battery.models import CNN, ConvType
from qmodem.tracking import get_run_parameters, retrieve_mlflow_setup_train
from qmodem.utils import setup_script_logging


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()

    hp = TestHyperparameters()

    mlflow_setup = retrieve_mlflow_setup_train()
    run_parameters = get_run_parameters(mlflow_setup.run_id, mlflow_setup.backend_store)

    model = CNN(
        conv_type=ConvType.DETERMINISTIC,
        in_features=1,
        n_filters=int(run_parameters["conv_n_filters"]),
        kernel_size=int(run_parameters["conv_kernel_size"]),
        dropout_rate=float(run_parameters["dropout_rate"]),
        act_fn=getattr(nnx, run_parameters["activation_function"]),
        rngs=nnx.Rngs(0),
    )  # RNGs won't be used for inference, so the seed is arbitrary.

    run_evaluation(
        model=model,
        hp=hp,
        mlflow_setup=mlflow_setup,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
        log_stream=log_stream,
        train_mode=True,  # Enables MC Dropout.
    )


if __name__ == "__main__":
    main()
