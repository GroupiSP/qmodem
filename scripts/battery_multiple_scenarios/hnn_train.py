from __future__ import annotations

import dataclasses
import functools
import os
import pathlib

import numpy as np
import sklearn.preprocessing as skpp
from dotenv import load_dotenv
from flax import nnx

from qmodem.battery.models import CNN, ConvType
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
        window_size=10,
        conv_kernel_size=3,
        conv_n_filters=29,
        beta_nll=0.14,
        learning_rate=0.00052,
        dropout_rate=0.315,
        early_stopping_patience=50,
        normalize_rul=True,
    )

    mlflow_setup = MLFlowSetup(run_name="hnn")

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

    model = CNN(
        conv_type=ConvType.DETERMINISTIC,
        in_features=2,
        n_filters=hp.conv_n_filters,
        kernel_size=hp.conv_kernel_size,  # Integer kernel size means 1D convolution, tuple means 2D convolution
        dropout_rate=hp.dropout_rate,
        act_fn=getattr(nnx, hp.activation_function),
        rngs=nnx.Rngs(hp.net_init_seed),
    )

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
            hyperparameters=dataclasses.asdict(hp),
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
