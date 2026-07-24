from __future__ import annotations

import dataclasses
import functools
import os
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import mlflow
import optax
import sklearn.preprocessing as skpp
from dotenv import load_dotenv

from qmodem.battery.models import HeteroscedasticCNN
from qmodem.data import (
    DataFrameSource,
    DataPipeline,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    normalize_ruls,
    to_jax,
)
from qmodem.module import nll_batched
from qmodem.tracking import (
    MLFlowSetup,
    track_mlflow,
)
from qmodem.train import (
    LogReporter,
    mlflow_track_losses,
    train_loop,
)
from qmodem.train_base import (
    EarlyStopper,
    mlflow_track_model_best_state,
)
from qmodem.utils import count_parameters, setup_script_logging
from scripts.battery.commons import (
    TrainHyperparameters,
    get_dataframes,
    train_dataloader_builder,
)


def main() -> None:
    load_dotenv()

    log_stream = setup_script_logging()

    hp = TrainHyperparameters()

    RAW_DATA_DIR = pathlib.Path(os.environ["RAW_DATA_DIR"])

    mlflow_setup = MLFlowSetup(
        run_name="hnn",
        experiment_name="checkpoint_phme_2026",
        run_description="""Baseline.""",
        tags={
            "model": "HNN",
            "case_study": "battery",
            "stage": "publication",
        },
    )

    # Model, schedule, optimizer
    model = HeteroscedasticCNN(rngs=nnx.Rngs(hp.net_init_seed))

    # Build the data sources, including windowing and normalization
    scaler = skpp.MinMaxScaler(feature_range=(0, 1))
    data_pipeline_train = DataPipeline(
        [
            functools.partial(
                get_time_windows_and_join,
                window_size=hp.window_size,
                stride=hp.stride,
            ),
            add_feature_dimension_to_y,
            functools.partial(normalize_ruls, transform_fn=scaler.fit_transform)
            if hp.normalize_rul
            else lambda x: x,
            to_jax,
        ]
    )
    data_pipeline_val = DataPipeline(
        [
            functools.partial(
                get_time_windows_and_join,
                window_size=hp.window_size,
                stride=hp.stride,
            ),
            add_feature_dimension_to_y,
            functools.partial(normalize_ruls, transform_fn=scaler.transform)
            if hp.normalize_rul
            else lambda x: x,
            to_jax,
        ]
    )

    data_gen_run = mlflow.get_run(os.environ["DATA_GEN_RUN_ID"])
    train_df, val_df, _ = get_dataframes(
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "test.csv",
        n_histories_train=int(data_gen_run.data.params["n_histories_train"]),
    )

    ds_train = DataFrameSource(df=train_df, pipeline=data_pipeline_train)
    ds_val = DataFrameSource(df=val_df, pipeline=data_pipeline_val)

    @nnx.jit
    def train_step(
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(model):
            return jnp.mean(nll_batched(model, batch, keys, beta=hp.beta_nll))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer = None,  # not used, but we keep the same signature as train_step for simplicity
    ) -> jax.Array:
        return jnp.mean(nll_batched(model, batch, keys, beta=hp.beta_nll))

    schedule = optax.cosine_decay_schedule(
        init_value=hp.learning_rate,
        decay_steps=hp.n_epochs * (len(ds_train) // hp.batch_size),
        alpha=hp.scheduler_alpha,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    early_stopper = EarlyStopper(
        patience=hp.early_stopping_patience, min_delta=hp.early_stopping_min_delta
    )

    with track_mlflow(setup=mlflow_setup):
        mlflow.sklearn.log_model(scaler, artifact_path="sklearn_scaler")
        mlflow.log_params(dataclasses.asdict(hp))
        mlflow.log_param("n_params", count_parameters(model))

        train_loop(
            n_epochs=hp.n_epochs,
            train_dataloader_builder=functools.partial(
                train_dataloader_builder,
                ds_train=ds_train,
                batch_size=hp.batch_size,
                drop_remainder=hp.drop_remainder,
            ),
            val_dataloader_builder=lambda n: [
                (ds_val.X, ds_val.y)
            ],  # single "batch" = whole val set, because no SGD happens at eval time
            initial_key=jax.random.key(hp.train_rng_seed),
            model=model,
            optimizer=optimizer,
            train_batch_fn=train_step,
            eval_batch_fn=eval_step,
            callbacks=[
                LogReporter(log_every=10),
                mlflow_track_model_best_state,
                mlflow_track_losses,
            ],
            early_stopper=early_stopper,
        )

        mlflow.log_text(log_stream.getvalue(), "training_log.txt")


if __name__ == "__main__":
    main()
