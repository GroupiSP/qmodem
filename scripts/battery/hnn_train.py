from __future__ import annotations

import dataclasses
import functools
import io
import logging
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import mlflow
import optax
import sklearn.preprocessing as skpp

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
    OutputVarianceTracker,
    mlflow_track_model_best_state,
)
from qmodem.utils import count_parameters
from scripts.battery.commons import (
    TrainHyperparameters,
    get_dataframes,
    train_dataloader_builder,
)
from scripts.battery.hnn_model import Net


def main() -> None:
    log_stream = io.StringIO()
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        handlers=[
            logging.StreamHandler(),  # console (stderr)
            logging.StreamHandler(log_stream),  # in-memory stream for MLflow logging
        ],
    )

    hp = TrainHyperparameters(early_stopping_patience=20)

    RAW_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    mlflow_setup = MLFlowSetup(
        run_name="hnn-8",
        experiment_name="one_key_one_datapoint",
        run_description="""1. Reseed the data sampler at every epoch, to avoid overfitting to the same data order
        \n2. Increase patience for early stopping,
        \n3. Fix labels/predictions shape mismatch in nll_batched,
        \n4. Add output variance callback (should be constant).""",
        tags={
            "model": "HNN",
            "case_study": "battery",
            "stage": "prototyping",
        },
    )

    # Model, schedule, optimizer
    model = Net(rngs=nnx.Rngs(hp.net_init_seed))

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

    train_df, val_df, _ = get_dataframes(
        RAW_DATA_DIR / "train.csv", RAW_DATA_DIR / "test.csv"
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

        key = jax.random.key(hp.train_rng_seed)
        key, subkey = jax.random.split(key)

        batch_variance_tracking = ds_val[
            jax.random.choice(
                subkey, len(ds_val), shape=(hp.batch_size,), replace=False
            )
        ]

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
            initial_key=key,
            model=model,
            optimizer=optimizer,
            train_batch_fn=train_step,
            eval_batch_fn=eval_step,
            callbacks=[
                LogReporter(log_every=10),
                mlflow_track_model_best_state,
                mlflow_track_losses,
                OutputVarianceTracker(
                    base_key=subkey,
                    X_batch=batch_variance_tracking[0],
                    n_samples=hp.n_samples_predictive_mean_variance,
                ),
            ],
            early_stopper=early_stopper,
        )

        mlflow.log_text(log_stream.getvalue(), "training_log.txt")


if __name__ == "__main__":
    main()
