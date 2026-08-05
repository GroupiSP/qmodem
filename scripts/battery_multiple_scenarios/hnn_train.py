from __future__ import annotations

import os
import pathlib

import flax.nnx as nnx
from dotenv import load_dotenv

from qmodem.battery.models import CNN, ConvType
from qmodem.battery.train import TrainHyperparameters, run_training
from qmodem.tracking import MLFlowSetup
from qmodem.utils import setup_script_logging


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()
    hp = TrainHyperparameters(
        conv_kernel_size=3,
        conv_n_filters=37,
        window_size=13,
        beta_nll=0.0207,
        learning_rate=0.008,
        dropout_rate=0.68,
    )

    mlflow_setup = MLFlowSetup(run_name="hnn_train_optimized_hps")
    model = CNN(
        conv_type=ConvType.DETERMINISTIC,
        in_features=1,
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
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
        log_stream=log_stream,
    )


if __name__ == "__main__":
    main()
