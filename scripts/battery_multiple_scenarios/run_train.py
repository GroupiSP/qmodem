from __future__ import annotations

import functools
import os
import pathlib
from dataclasses import asdict

import numpy as np
import sklearn.preprocessing as skpp
from dotenv import load_dotenv

from qmodem.battery.dispatch import ModelBuildParameters, build_model
from qmodem.battery.tracking import log_general
from qmodem.battery.train import TrainHyperparameters, run_training
from qmodem.data import (
    DataPipeline,
    IdentityScaler,
    ScalingStep,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    to_jax,
)
from qmodem.tracking import MLFlowSetup, track_mlflow, write_setup_to_file
from qmodem.utils import count_parameters, setup_script_logging


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()
    hp = TrainHyperparameters(
        method="bnn",
        window_size=10,
        conv_kernel_size=3,
        conv_n_filters=29,
        beta_nll=0.002,
        learning_rate=0.0001,
        dropout_rate=0.0384,
        early_stopping_patience=50,
        normalize_rul=True,
    )

    mlflow_setup = MLFlowSetup(run_name="bnn")

    x_scaler = skpp.StandardScaler()
    y_scaler = (
        skpp.MinMaxScaler(feature_range=(0, 1))
        if hp.normalize_rul
        else IdentityScaler()
    )

    pipeline = DataPipeline(
        [
            ScalingStep(scaler=x_scaler, features=["load", "voltage"]),
            ScalingStep(scaler=y_scaler, features=["rul"]),
            functools.partial(
                get_time_windows_and_join,
                window_size=hp.window_size,
                stride=hp.stride,
                features=["load", "voltage"],
            ),
            add_feature_dimension_to_y,
            to_jax,
        ]
    )

    model = build_model(ModelBuildParameters(**asdict(hp)))

    with track_mlflow(mlflow_setup):
        write_setup_to_file()

        run_training(
            model=model,
            hp=hp,
            raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
            data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
            data_pipeline=pipeline,
        )

        log_general(
            num_model_params=count_parameters(model),
            hyperparameters=asdict(hp),
            scalers={
                "x_scaler": (
                    x_scaler,
                    "transform",
                    np.ones(shape=(hp.window_size, 1), dtype=np.float32),
                ),
                "y_scaler": (
                    y_scaler,
                    "inverse_transform",
                    np.ones(shape=(1, 1), dtype=np.float32),
                ),
            },
            log_stream=log_stream,
        )


if __name__ == "__main__":
    main()
