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
from qmodem.module import negative_log_likelihood
from qmodem.tracking import (
    MLFlowSetup,
    track_mlflow,
)
from qmodem.train import (
    EarlyStopper,
    LogReporter,
    mlflow_track_losses,
    mlflow_track_model_best_state,
    train_loop,
)
from qmodem.utils import count_parameters
from scripts.battery.bnn_model import Net
from scripts.battery.commons import (
    TrainHyperparameters,
    create_dataloaders,
    get_dataframes,
)


@dataclasses.dataclass
class Hyperparameters(TrainHyperparameters):
    pass


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

    hp = Hyperparameters()

    RAW_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    mlflow_setup = MLFlowSetup(
        run_name="bnn",
        experiment_name="phme26",
        tags={
            "model": "BNN",
            "case_study": "battery",
            "stage": "publishing",
            "publication": "phme26",
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

    # Dataloaders
    dataloader_train, dataloader_val = create_dataloaders(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=hp.batch_size,
        sampler_seeds=hp.sampler_seeds,
        drop_remainder=hp.drop_remainder,
    )

    # Loss evaluation functions and steps for the training.
    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
    def per_sample_nll(model, sample, sample_key) -> jax.Array:
        # Build the RNG here to avoid crossing different trace levels.
        rngs = nnx.Rngs(params=sample_key)
        return negative_log_likelihood(model, sample, rngs, beta=hp.beta_nll)

    def per_sample_kl(model) -> jax.Array:
        # KL divergence is deterministic, so we don't need to vmap over samples or use RNGs.
        return model.kl_divergence() / len(ds_train)  # average KL per sample

    def elbo_loss(model, batch, keys) -> jax.Array:
        # This is the ELBO loss for the batch.
        return jnp.mean(per_sample_nll(model, batch, keys)) + batch[0].shape[
            0
        ] * per_sample_kl(model)

    @nnx.jit
    def train_step(
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(model):
            # This is the ELBO loss for the batch.
            return elbo_loss(model, batch, keys)

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
        return elbo_loss(model, batch, keys)

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
            dataloader_train=dataloader_train,
            dataloader_val=dataloader_val,
            initial_key=jax.random.PRNGKey(hp.train_rng_seed),
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
