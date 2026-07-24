from __future__ import annotations

import os
import pathlib

import flax.nnx as nnx
from dotenv import load_dotenv

from qmodem.battery.models import MCDropoutCNN
from qmodem.battery.train import MCDTrainHyperparameters, run_training
from qmodem.tracking import MLFlowSetup
from qmodem.utils import setup_script_logging


def main() -> None:
    load_dotenv()
    log_stream = setup_script_logging()
    hp = MCDTrainHyperparameters(early_stopping_patience=20)

    mlflow_setup = MLFlowSetup(
        run_name="mcd",
        experiment_name="checkpoint_phme_2026",
        run_description="""Baseline.""",
        tags={
            "model": "MCD",
            "case_study": "battery",
            "stage": "publication",
        },
    )
    model = MCDropoutCNN(
        n_filters=hp.conv_n_filters,
        kernel_size=hp.conv_kernel_size,
        dropout_rate=hp.dropout_rate,
        act_fn=getattr(nnx, hp.activation_function),
        rngs=nnx.Rngs(hp.net_init_seed),
    )

    run_training(
        model=model,
        hp=hp,
        mlflow_setup=mlflow_setup,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID"],
        log_stream=log_stream,
    )


if __name__ == "__main__":
    main()
