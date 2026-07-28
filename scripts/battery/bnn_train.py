from __future__ import annotations

import os
import pathlib

import flax.nnx as nnx
from dotenv import load_dotenv

from qmodem.battery.models import BayesianCNN
from qmodem.battery.train import TrainHyperparameters, run_training
from qmodem.battery.train_steps import make_elbo_steps
from qmodem.callbacks import track_conv_weights_variance
from qmodem.tracking import MLFlowSetup
from qmodem.utils import setup_script_logging


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()
    hp = TrainHyperparameters()

    mlflow_setup = MLFlowSetup(run_name="dummy_bnn")
    model = BayesianCNN(
        n_filters=hp.conv_n_filters,
        kernel_size=hp.conv_kernel_size,
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
        step_factory=make_elbo_steps,
        callbacks=(track_conv_weights_variance,),
    )


if __name__ == "__main__":
    main()
