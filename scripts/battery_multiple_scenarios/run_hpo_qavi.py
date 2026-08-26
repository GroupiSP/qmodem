from __future__ import annotations

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

from qmodem.battery.data_generation import (
    load_simulation_config,
    reconstruct_true_rul_distribution,
    sim_updater_two_scenarios,
)
from qmodem.battery.dispatch import build_discriminator, build_qavi_model
from qmodem.battery.evaluate import run_evaluation
from qmodem.battery.hpo import (
    QAVIHPOHyperparameters,
    get_average_crps,
    get_validation_data,
)
from qmodem.battery.tracking import log_general
from qmodem.battery.train import QAVITrainHyperparameters, run_adversarial_training
from qmodem.battery.train_steps import make_qavi_steps
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


def objective_factory(
    hp_hpo: QAVIHPOHyperparameters,
) -> Callable[[optuna.Trial], float]:
    """Creates an Optuna objective function for QAVI hyperparameter optimisation.

    Args:
        hp_hpo: QAVI HPO hyperparameters defining the search space bounds.

    Returns:
        An objective function that, given an Optuna trial, trains a QAVI model and
        returns the average CRPS on the validation set.
    """

    def objective(trial: optuna.Trial) -> float:
        window_size = trial.suggest_int(
            "window_size", hp_hpo.window_size_min, hp_hpo.window_size_max, step=5
        )
        max_kernel_size = (window_size - 1) // 3
        kernel_size_upper = max(
            hp_hpo.kernel_size_min, min(hp_hpo.kernel_size_ceil, max_kernel_size)
        )
        conv_kernel_size = trial.suggest_int(
            "conv_kernel_size",
            hp_hpo.kernel_size_min,
            kernel_size_upper,
        )

        hp_train = QAVITrainHyperparameters(
            conv_kernel_size=conv_kernel_size,
            conv_n_filters=trial.suggest_int(
                "conv_n_filters", hp_hpo.conv_n_filters_min, hp_hpo.conv_n_filters_max
            ),
            window_size=window_size,
            beta_nll=trial.suggest_float(
                "beta_nll", hp_hpo.beta_nll_min, hp_hpo.beta_nll_max
            ),
            dropout_rate=trial.suggest_float(
                "dropout_rate", hp_hpo.dropout_rate_min, hp_hpo.dropout_rate_max
            ),
            pqc_n_qubits=trial.suggest_int(
                "pqc_n_qubits", hp_hpo.pqc_n_qubits_min, hp_hpo.pqc_n_qubits_max
            ),
            pqc_n_layers=trial.suggest_int(
                "pqc_n_layers", hp_hpo.pqc_n_layers_min, hp_hpo.pqc_n_layers_max
            ),
            learning_rate_generator=trial.suggest_float(
                "learning_rate_generator",
                hp_hpo.lr_generator_min,
                hp_hpo.lr_generator_max,
                log=True,
            ),
            learning_rate_discriminator=trial.suggest_float(
                "learning_rate_discriminator",
                hp_hpo.lr_discriminator_min,
                hp_hpo.lr_discriminator_max,
                log=True,
            ),
            adversarial_loss_weight=trial.suggest_float(
                "adversarial_loss_weight",
                hp_hpo.adversarial_loss_weight_min,
                hp_hpo.adversarial_loss_weight_max,
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

        model = build_qavi_model(hp_train)
        discriminator = build_discriminator(hp_train)

        generator_step, discriminator_step, eval_step = make_qavi_steps(
            beta=hp_train.beta_nll,
            adversarial_loss_weight=hp_train.adversarial_loss_weight,
        )

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)

            run_adversarial_training(
                model=model,
                discriminator=discriminator,
                hp=hp_train,
                raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
                data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
                data_pipeline=pipeline,
                generator_batch_fn=generator_step,
                discriminator_batch_fn=discriminator_step,
                eval_batch_fn=eval_step,
            )

            log_general(
                num_model_params=count_parameters(model),
                hyperparameters=hp_train.model_dump(),
                scalers={
                    "x_scaler": (
                        x_scaler,
                        "transform",
                        np.ones(shape=(hp_train.window_size, 2), dtype=np.float32),
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
    hp = QAVIHPOHyperparameters()

    mlflow_setup = MLFlowSetup(run_name="qavi_hpo")

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
        mlflow.log_params(best_trial.params)

        mlflow.log_text(log_stream.getvalue(), "hpo_log.txt")


if __name__ == "__main__":
    main()
