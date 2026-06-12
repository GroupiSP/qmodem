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
from qmodem.tracking import (
    MLFlowSetup,
    mlflow_track_losses,
    mlflow_track_model_best_state,
    track_mlflow,
)
from qmodem.train_adversarial import (
    EarlyStopper,
    LogReporter,
    train_loop,
)
from qmodem.utils import count_parameters
from scripts.battery.commons import (
    TrainHyperparameters,
    create_dataloaders,
    get_dataframes,
)
from scripts.battery.qavi_model import Net


class Discriminator(nnx.Module):
    """MLP discriminator: input_dim → hidden → hidden → 1."""

    def __init__(self, input_dim: int, hidden: int = 64, *, rngs: nnx.Rngs) -> None:
        self.l1 = nnx.Linear(input_dim, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        x = x.squeeze(-1)  # Removes the feature dimension
        x = nnx.leaky_relu(self.l1(x), negative_slope=0.2)
        x = nnx.leaky_relu(self.l2(x), negative_slope=0.2)
        return nnx.sigmoid(self.l3(x))


@dataclasses.dataclass
class Hyperparameters(TrainHyperparameters):
    pqc_n_qubits: int = 5
    pqc_n_layers: int = 1
    discriminator_hidden_size: int = 64
    discriminator_act_fn: str = "leaky_relu"
    discriminator_init_seed: int = 43
    learning_rate: None = None  # override
    learning_rate_generator: float = 1e-3
    learning_rate_discriminator: float = 1e-3
    early_stopping_patience: int = 30  # override
    scheduler_alpha: None = None  # override


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
        run_name="qavi",
        experiment_name="battery_default",
        tags={
            "model": "QAVI",
            "case_study": "battery",
            "stage": "prototyping",
            "publication": "phme26",
        },
    )

    # Model, schedule, optimizer
    model = Net(rngs=nnx.Rngs(hp.net_init_seed))
    discriminator = Discriminator(
        input_dim=hp.window_size
        + 1,  # +1 for the RUL value concatenated to the input window
        hidden=hp.discriminator_hidden_size,
        rngs=nnx.Rngs(hp.discriminator_init_seed),
    )

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

    @nnx.jit
    def discriminator_step(
        model: nnx.Module,
        discriminator: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(discriminator):
            # Build the RNG here to avoid crossing different trace levels.
            eps = 1e-8

            x, y_true = batch
            rngs = nnx.Rngs(params=keys[0])

            y_pred = model(x, rngs)  # (1, 2) -> mu, var
            mu_pred = y_pred[:, :1]  # (1,1)

            proba_real = discriminator(
                jnp.concatenate([x, y_true[:, :, None]], axis=1), rngs
            )
            proba_fake = discriminator(
                jnp.concatenate([x, mu_pred[:, :, None]], axis=1), rngs
            )
            error = -jnp.log(proba_real + eps) - jnp.log(1 - proba_fake + eps)

            return jnp.mean(error.squeeze(-1))

        loss, grads = nnx.value_and_grad(loss_fn)(discriminator)
        optimizer.update(discriminator, grads)
        return loss

    @nnx.jit
    def generator_step(
        model: nnx.Module,
        discriminator: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(model):
            eps = 1e-8

            xs, y_true = batch  # xs: (batch, ...), y_true: (batch, 1)
            rngs = nnx.Rngs(
                params=keys[0]
            )  # one key suffices — weights are batch-shared

            # PQC generates weights once, conv applied to whole batch
            y_pred = model(xs, rngs)  # (batch, 2)
            mu_pred = y_pred[:, :1]  # (batch, 1)

            proba_fake = discriminator(
                jnp.concatenate([xs, mu_pred[:, :, None]], axis=1), rngs
            )  # (batch, 1)
            proba_fake_clipped = jnp.clip(proba_fake, eps, 1 - eps)
            logits = jnp.log(proba_fake_clipped / (1 - proba_fake_clipped))
            adv_error = -logits.squeeze(-1)  # (batch,)

            # NLL per sample from already-computed predictions, no second model call
            mu_pred_1d = y_pred[:, 0]
            variances_pred_1d = jnp.clip(y_pred[:, 1], min=1e-8)
            y_true_1d = y_true.squeeze(-1)
            nll = (
                0.5 * jnp.log(variances_pred_1d)
                + 0.5 * jnp.square(y_true_1d - mu_pred_1d) / variances_pred_1d
            )  # (batch,)

            return jnp.mean(adv_error + nll)

        # Notice that vmapping does not work in this case, because of an
        # incompatibility downstream with the PenyyLane qnode.
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
        def loss_fn(model):
            xs, y_true = batch  # xs: (batch, ...), y_true: (batch, 1)
            rngs = nnx.Rngs(params=keys[0])

            y_pred = model(xs, rngs)  # (batch, 2)
            mu_pred_1d = y_pred[:, 0]  # (batch,)
            variances_pred_1d = jnp.clip(y_pred[:, 1], min=1e-8)  # (batch,)
            y_true_1d = y_true.squeeze(-1)  # (batch,)

            nll = (
                0.5 * jnp.log(variances_pred_1d)
                + 0.5 * jnp.square(y_true_1d - mu_pred_1d) / variances_pred_1d
            )  # (batch,)

            return jnp.mean(nll)

        return loss_fn(model)

    optimizer_discriminator = nnx.Optimizer(
        discriminator, optax.adam(hp.learning_rate_discriminator), wrt=nnx.Param
    )
    optimizer_generator = nnx.Optimizer(
        model, optax.adam(hp.learning_rate_generator), wrt=nnx.Param
    )

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
            initial_key=jax.random.key(hp.train_rng_seed),
            model=model,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            generator_batch_fn=generator_step,
            discriminator_batch_fn=discriminator_step,
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
