from __future__ import annotations

import os
import pathlib

import flax.nnx as nnx
import optuna
from dotenv import load_dotenv

from qmodem.battery.hpo import HPOHyperparameters, score_avg_val_crps
from qmodem.battery.models import HeteroscedasticCNN
from qmodem.battery.train import TrainHyperparameters, run_training
from qmodem.tracking import MLFlowSetup
from qmodem.utils import setup_script_logging


def objective(trial: optuna.Trial) -> float:
    load_dotenv(override=True)

    log_stream = setup_script_logging()
    hp_train = TrainHyperparameters()
    hp_objective = HPOHyperparameters()

    mlflow_setup = MLFlowSetup(run_name="dummy_hnn_multiple")
    model = HeteroscedasticCNN(
        n_filters=hp_train.conv_n_filters,
        kernel_size=hp_train.conv_kernel_size,
        act_fn=getattr(nnx, hp_train.activation_function),
        rngs=nnx.Rngs(hp_train.net_init_seed),
    )

    run_training(
        model=model,
        hp=hp_train,
        mlflow_setup=mlflow_setup,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
        log_stream=log_stream,
    )

    return score_avg_val_crps(
        model=model,
        mlflow_setup=mlflow_setup,
        hp=hp_objective,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        validation_history_ids=list(range(50, 70)),
        window_size=hp_train.window_size,
    )


def main() -> None:
    pass


if __name__ == "__main__":
    main()
