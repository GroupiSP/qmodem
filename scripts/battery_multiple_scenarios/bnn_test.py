from __future__ import annotations

import os
import pathlib

import flax.nnx as nnx
import mlflow
from dotenv import load_dotenv

from qmodem.battery.evaluate import Hyperparameters, run_evaluation
from qmodem.battery.models import BayesianCNN
from qmodem.tracking import MLFlowSetup
from qmodem.utils import setup_script_logging


def main() -> None:
    load_dotenv()

    log_stream = setup_script_logging()

    TRAIN_RUN_ID = "a7145df57fed4061a8eaf82f38c6b049"

    # TODO: set the **default** number of test SOC_0s to 20.
    hp = Hyperparameters(test_n_soc0s=20)

    mlflow_setup = MLFlowSetup(
        experiment_name="variable_loading_conditions", run_id=TRAIN_RUN_ID
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
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
        log_stream=log_stream,
        train_mode=False,
    )


if __name__ == "__main__":
    main()
