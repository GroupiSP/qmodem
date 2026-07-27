from __future__ import annotations

import os
import pathlib

import flax.nnx as nnx
from dotenv import load_dotenv

from qmodem.battery.models import Discriminator, QuantumVICNN, WeightGenerator
from qmodem.battery.train import QAVITrainHyperparameters, run_adversarial_training
from qmodem.battery.train_steps import make_qavi_steps
from qmodem.tracking import MLFlowSetup
from qmodem.utils import setup_script_logging


def main() -> None:
    load_dotenv(override=True)
    log_stream = setup_script_logging()
    hp = QAVITrainHyperparameters(
        pqc_n_layers=2,
        early_stopping_patience=500,
    )

    mlflow_setup = MLFlowSetup(
        run_name="qavi",
        experiment_name="checkpoint_phme_2026",
        tags={
            "model": "QAVI",
            "case_study": "battery",
            "stage": "prototyping",
        },
        run_description="""Baseline.""",
    )

    weight_generator = WeightGenerator(
        n_qubits=hp.pqc_n_qubits,
        n_layers=hp.pqc_n_layers,
        kernel_size=hp.conv_kernel_size,
        in_features=1,
        out_features=hp.conv_n_filters,
    )
    model = QuantumVICNN(
        n_filters=hp.conv_n_filters,
        kernel_size=hp.conv_kernel_size,
        generator=weight_generator,
        act_fn=getattr(nnx, hp.activation_function),
        rngs=nnx.Rngs(hp.net_init_seed),
    )
    discriminator = Discriminator(
        input_dim=hp.window_size + 1,
        hidden=hp.discriminator_hidden_size,
        rngs=nnx.Rngs(hp.discriminator_init_seed),
    )
    generator_step, discriminator_step, eval_step = make_qavi_steps(
        beta=hp.beta_nll,
        adversarial_loss_weight=hp.adversarial_loss_weight,
    )

    run_adversarial_training(
        model=model,
        discriminator=discriminator,
        hp=hp,
        mlflow_setup=mlflow_setup,
        raw_data_dir=pathlib.Path(os.environ["RAW_DATA_DIR"]),
        data_gen_run_id=os.environ["DATA_GEN_RUN_ID"],
        log_stream=log_stream,
        generator_batch_fn=generator_step,
        discriminator_batch_fn=discriminator_step,
        eval_batch_fn=eval_step,
    )


if __name__ == "__main__":
    main()
