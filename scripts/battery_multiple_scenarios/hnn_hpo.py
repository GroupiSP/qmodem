from __future__ import annotations

import dataclasses
import functools
import io
import logging
import os
import pathlib

import mlflow
import numpy as np
import optuna
import sklearn.preprocessing as skpp
from dotenv import load_dotenv
from flax import nnx

from qmodem.battery.hpo import HPOHyperparameters, score_avg_val_crps
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
from qmodem.tracking import MLFlowSetup, track_mlflow
from qmodem.utils import count_parameters, setup_script_logging


def objective_factory(hp_hpo: HPOHyperparameters, log_stream: io.StringIO) -> float:
    def objective(trial: optuna.Trial) -> float:
        # TODO Move hyperparameters of the HPO to `hpo.py`
        window_size = trial.suggest_int("window_size", 50, 350, step=25)
        max_kernel_size = (window_size - 1) // 3
        conv_kernel_size = trial.suggest_int(
            "conv_kernel_size", 5, min(50, max_kernel_size)
        )

        hp_train = TrainHyperparameters(
            conv_kernel_size=conv_kernel_size,
            conv_n_filters=trial.suggest_int("conv_n_filters", 4, 40),
            window_size=window_size,
            beta_nll=trial.suggest_float("beta_nll", 0.0, 1.0),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            dropout_rate=trial.suggest_float("dropout_rate", 0.0, 0.9),
        )

        x_scaler = skpp.StandardScaler()
        y_scaler = (
            skpp.MinMaxScaler(feature_range=(0, 1))
            if hp_train.normalize_rul
            else IdentityScaler()
        )

        pipeline = DataPipeline(
            [
                ScalingStep(scaler=x_scaler, features=["load", "voltage"]),
                ScalingStep(scaler=y_scaler, features=["rul"]),
                functools.partial(
                    get_time_windows_and_join,
                    window_size=hp_train.window_size,
                    stride=hp_train.stride,
                    features=["load", "voltage"],
                ),
                add_feature_dimension_to_y,
                to_jax,
            ]
        )

        model = CNN(
            conv_type=ConvType.DETERMINISTIC,
            in_features=2,
            n_filters=hp_train.conv_n_filters,
            kernel_size=hp_train.conv_kernel_size,
            dropout_rate=hp_train.dropout_rate,
            act_fn=getattr(nnx, hp_train.activation_function),
            rngs=nnx.Rngs(hp_train.net_init_seed),
        )

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)

            run_training(
                model=model,
                hp=hp_train,
                raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
                data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
                data_pipeline=pipeline,
            )

            log_general(
                num_model_params=count_parameters(model),
                hyperparameters=dataclasses.asdict(hp_train),
                scalers={
                    "x_scaler": (
                        x_scaler,
                        "transform",
                        np.ones(shape=(hp_train.window_size, 1), dtype=np.float32),
                    ),
                    "y_scaler": (
                        y_scaler,
                        "inverse_transform",
                        np.ones(shape=(1, 1), dtype=np.float32),
                    ),
                },
            )

            return score_avg_val_crps(
                model=model,
                hp=hp_hpo,
                raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
                validation_history_ids=list(range(10, 15)),
                features=["load", "voltage"],
            )

    return objective


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()
    hp = HPOHyperparameters()

    mlflow_setup = MLFlowSetup(run_name="hnn_hpo")

    hp_sampler = optuna.samplers.TPESampler(seed=hp.seed_hp_sampler)
    study = optuna.create_study(sampler=hp_sampler, direction="minimize")

    with track_mlflow(mlflow_setup):
        study.optimize(
            func=objective_factory(hp, log_stream),
            n_trials=hp.num_hp_trials,
        )

        logging.info("Best trial:")
        best_trial = study.best_trial

        logging.info(f"Value: {best_trial.value}")
        logging.info("Params: ")
        for key, value in best_trial.params.items():
            logging.info(f"{key}: {value}")

        mlflow.log_param("best_trial_id", best_trial.number)
        mlflow.log_params(best_trial.params)  # log best trial params to the parent run

        mlflow.log_text(log_stream.getvalue(), "hpo_log.txt")


if __name__ == "__main__":
    main()
