from __future__ import annotations

import functools
import os
import pathlib
from dataclasses import asdict

import numpy as np
import pennylane as qp
import sklearn.preprocessing as skpp
from dotenv import load_dotenv
from flax import nnx

from qmodem.battery.models import (
    CNN,
    ContinuousWeightsGenerator,
    ConvType,
    Discriminator,
)
from qmodem.battery.tracking import log_general
from qmodem.battery.train import TrainHyperparameters, run_adversarial_training
from qmodem.battery.train_steps import make_qavi_steps
from qmodem.data import (
    DataPipeline,
    IdentityScaler,
    ScalingStep,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    to_jax,
)
from qmodem.quantum_circuits import ContinuousCircuitFactory
from qmodem.tracking import MLFlowSetup, track_mlflow, write_setup_to_file
from qmodem.utils import count_parameters, setup_script_logging


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()
    hp = TrainHyperparameters(
        method="qavi",
        conv_kernel_size=3,
        conv_n_filters=4,
        window_size=10,
        beta_nll=0.5,
        dropout_rate=0.1,
        pqc_n_qubits=6,
        pqc_n_layers=2,
    )
    # TODO: Add in_features to hyperparameters

    mlflow_setup = MLFlowSetup(run_name="qavi")

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

    circuit_factory = ContinuousCircuitFactory(
        n_qubits=hp.pqc_n_qubits, n_layers=hp.pqc_n_layers
    )

    device = qp.device("default.qubit", wires=hp.pqc_n_qubits)

    weight_generator = ContinuousWeightsGenerator(
        circuit_factory=circuit_factory,
        device=device,
        kernel_size=hp.conv_kernel_size,
        in_features=2,
        out_features=hp.conv_n_filters,
    )
    model = CNN(
        conv_type=ConvType.QUANTUM_GENERATED,
        in_features=2,
        n_filters=hp.conv_n_filters,
        kernel_size=hp.conv_kernel_size,
        generator=weight_generator,
        act_fn=getattr(nnx, hp.activation_function),
        rngs=nnx.Rngs(hp.net_init_seed),
    )

    discriminator = Discriminator(
        input_dim=2 * hp.window_size + 1,  # 2 channels (current and voltage) + 1 RUL
        hidden=hp.discriminator_hidden_size,
        rngs=nnx.Rngs(hp.discriminator_init_seed),
    )
    generator_step, discriminator_step, eval_step = make_qavi_steps(
        beta=hp.beta_nll,
        adversarial_loss_weight=hp.adversarial_loss_weight,
    )

    with track_mlflow(mlflow_setup):
        write_setup_to_file()

        run_adversarial_training(
            model=model,
            discriminator=discriminator,
            hp=hp,
            raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
            data_gen_run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
            data_pipeline=pipeline,
            generator_batch_fn=generator_step,
            discriminator_batch_fn=discriminator_step,
            eval_batch_fn=eval_step,
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
