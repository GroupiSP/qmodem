from __future__ import annotations

import dataclasses
import functools
import logging
import os
import pathlib
from collections.abc import Callable

import jax
import mlflow
import numpy as np
import optuna
import sklearn.preprocessing as skpp
from dotenv import load_dotenv
from flax import nnx

from qmodem.battery.data_generation import (
    load_simulation_config,
    reconstruct_true_rul_distribution,
    sim_updater_two_scenarios,
)
from qmodem.battery.evaluate import run_evaluation
from qmodem.battery.hpo import (
    HPOHyperparameters,
    get_average_crps,
    get_validation_data,
)
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

logger = logging.getLogger(__name__)


def objective_factory(hp_hpo: HPOHyperparameters) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        # TODO Move hyperparameters of the HPO to `hpo.py`
        window_size = trial.suggest_int(
            "window_size", hp_hpo.window_size_min, hp_hpo.window_size_max, step=5
        )
        max_kernel_size = (window_size - 1) // 3
        conv_kernel_size = trial.suggest_int(
            "conv_kernel_size",
            hp_hpo.kernel_size_min,
            min(hp_hpo.kernel_size_ceil, max_kernel_size),
        )

        hp_train = TrainHyperparameters(
            method="mcd",
            conv_kernel_size=conv_kernel_size,
            conv_n_filters=trial.suggest_int(
                "conv_n_filters", hp_hpo.conv_n_filters_min, hp_hpo.conv_n_filters_max
            ),
            window_size=window_size,
            beta_nll=trial.suggest_float(
                "beta_nll", hp_hpo.beta_nll_min, hp_hpo.beta_nll_max
            ),
            learning_rate=trial.suggest_float(
                "learning_rate", hp_hpo.lr_min, hp_hpo.lr_max, log=True
            ),
            dropout_rate=trial.suggest_float(
                "dropout_rate", hp_hpo.dropout_rate_min, hp_hpo.dropout_rate_max
            ),
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
            mcd=hp_train.method == "mcd",
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

            # Start evaluation
            model.eval()

            # Load the test raw data
            raw_data_dir = pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"])
            validation_data = get_validation_data(
                raw_data_dir / "train.csv", list(range(10, 15))
            )

            simulator_base_config = load_simulation_config(
                run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
            )

            true_rul_samples = reconstruct_true_rul_distribution(
                test_data=validation_data,
                simulator_base_config=simulator_base_config,
                n_soc0s=hp_hpo.num_soc0s_eval,
                n_mc_samples=hp_hpo.num_mc_samples,
                simulator_updater=sim_updater_two_scenarios,
                event_fn=lambda t0, soc0: t0 > 900.0,
            )

            key = jax.random.key(hp_hpo.seed_objective)

            validation_case_results = run_evaluation(
                model=model,
                test_data=validation_data,
                test_rul_samples=true_rul_samples,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                window_size=window_size,
                n_soc0s=hp_hpo.num_soc0s_eval,
                n_mc_samples=hp_hpo.num_mc_samples,
                features=["load", "voltage"],
                key=key,
            )

            rul_grid = np.arange(
                hp_hpo.rul_grid_crps_start,
                hp_hpo.rul_grid_crps_end,
                hp_hpo.rul_grid_crps_resolution,
            )

            avg_crps = get_average_crps(validation_case_results, rul_grid)

            mlflow.log_metric("avg_crps", avg_crps)
            return avg_crps

    return objective


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()
    hp = HPOHyperparameters()

    mlflow_setup = MLFlowSetup(run_name="mcd_hpo")

    hp_sampler = optuna.samplers.TPESampler(seed=hp.seed_hp_sampler)
    study = optuna.create_study(sampler=hp_sampler, direction="minimize")

    with track_mlflow(mlflow_setup):
        study.optimize(
            func=objective_factory(hp),
            n_trials=hp.num_hp_trials,
        )

        logger.info("Best trial:")
        best_trial = study.best_trial

        logger.info(f"Value: {best_trial.value}")
        logger.info("Params: ")
        for key, value in best_trial.params.items():
            logger.info(f"{key}: {value}")

        mlflow.log_param("best_trial_id", best_trial.number)
        mlflow.log_params(best_trial.params)  # log best trial params to the parent run

        mlflow.log_text(log_stream.getvalue(), "hpo_log.txt")


if __name__ == "__main__":
    main()
