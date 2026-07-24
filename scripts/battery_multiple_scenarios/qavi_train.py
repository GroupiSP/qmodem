from __future__ import annotations

import dataclasses
import functools
import io
import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import mlflow
import optax
import sklearn.preprocessing as skpp

from qmodem.battery.models import QuantumVICNN, WeightGenerator
from qmodem.data import (
    DataFrameSource,
    DataPipeline,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    normalize_ruls,
    to_jax,
)
from qmodem.module import model_fwd, nll_batched
from qmodem.tracking import (
    MLFlowSetup,
    track_mlflow,
)
from qmodem.train_adversarial import (
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
from scripts.battery_multiple_scenarios.commons import (
    DATA_GEN_RUN_ID,
    RAW_DATA_DIR,
    TrainHyperparameters,
    get_dataframes,
    train_dataloader_builder,
)


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
    learning_rate: None = None  # override (LRs are separate)
    learning_rate_generator: float = 1e-3
    learning_rate_discriminator: float = 1e-3
    scheduler_alpha: None = None  # override (No scheduler)


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

    hp = Hyperparameters(pqc_n_layers=2)

    mlflow_setup = MLFlowSetup(
        run_name="qavi",
        experiment_name="variable_loading_conditions",
        tags={
            "model": "QAVI",
            "case_study": "battery",
            "stage": "prototyping",
        },
        run_description="""Baseline.""",
    )

    # Generator of weights for the convolutional layer
    w_gen = WeightGenerator(
        n_qubits=hp.pqc_n_qubits,
        n_layers=hp.pqc_n_layers,
        kernel_size=hp.conv_kernel_size,
        in_features=1,
        out_features=hp.conv_n_filters,
    )

    # Model, schedule, optimizer
    model = QuantumVICNN(
        n_filters=hp.conv_n_filters,
        kernel_size=hp.conv_kernel_size,
        generator=w_gen,
        act_fn=getattr(nnx, hp.activation_function),
        rngs=nnx.Rngs(hp.net_init_seed),
    )
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

    data_gen_run = mlflow.get_run(DATA_GEN_RUN_ID)
    train_df, val_df, _ = get_dataframes(
        RAW_DATA_DIR / "train.csv",
        RAW_DATA_DIR / "test.csv",
        n_histories_train=int(data_gen_run.data.params["n_histories_train"]),
    )

    ds_train = DataFrameSource(df=train_df, pipeline=data_pipeline_train)
    ds_val = DataFrameSource(df=val_df, pipeline=data_pipeline_val)

    @nnx.jit
    def discriminator_step(
        model: nnx.Module,
        discriminator: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(discriminator, model):
            # Build the RNG here to avoid crossing different trace levels.
            eps = 1e-8

            # NOTE: squeezing the ys makes them of the same shape as the predicted labels.
            xs, ys_true = batch[0], batch[1].squeeze(-1)
            ns = len(xs)  # batch size
            key_0, key_1, key_2 = jax.random.split(key, num=3)

            model_keys = jax.random.split(key_0, num=ns)
            model_out = model_fwd(model, xs, model_keys)  # (1, 2) -> mu, var
            mu_pred, var_pred = model_out[:, 0], model_out[:, 1]  # (batch,), (batch,)

            # Sample the output normal distribution
            std_pred = jnp.sqrt(jnp.clip(var_pred, min=1e-8))  # Ensure std is positive
            y_pred = (
                jax.random.normal(key_1, shape=(ns,)) * std_pred + mu_pred
            )  # (batch,)

            rngs = nnx.Rngs(params=key_2)
            proba_real = discriminator(
                jnp.concatenate([xs, ys_true[:, None, None]], axis=1), rngs
            )
            proba_fake = discriminator(
                jnp.concatenate([xs, y_pred[:, None, None]], axis=1), rngs
            )
            error = -jnp.log(proba_real + eps) - jnp.log(1 - proba_fake + eps)

            return jnp.mean(error.squeeze(-1))

        loss, grads = nnx.value_and_grad(loss_fn)(discriminator, model)
        optimizer.update(discriminator, grads)
        return loss

    @nnx.jit
    def generator_step(
        model: nnx.Module,
        discriminator: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(model):
            eps = 1e-8

            # NOTE: squeezing the ys makes them of the same shape as the predicted labels.
            xs = batch[0]

            ns = len(xs)  # batch size
            key_0, key_1, key_2, key_3 = jax.random.split(key, num=4)

            model_keys = jax.random.split(key_0, num=ns)
            model_out = model_fwd(model, xs, model_keys)  # (1, 2) -> mu, var
            mu_pred, var_pred = model_out[:, 0], model_out[:, 1]  # (batch,), (batch,)

            # Sample the output normal distribution
            std_pred = jnp.sqrt(jnp.clip(var_pred, min=1e-8))  # Ensure std is positive
            y_pred = (
                jax.random.normal(key_1, shape=(ns,)) * std_pred + mu_pred
            )  # (batch,)

            rngs = nnx.Rngs(params=key_2)
            proba_fake = discriminator(
                jnp.concatenate([xs, y_pred[:, None, None]], axis=1), rngs
            )  # (batch, 1)
            proba_fake_clipped = jnp.clip(proba_fake, eps, 1 - eps)
            neg_log_proba_fake = -jnp.log(proba_fake_clipped)
            adv_error = neg_log_proba_fake.squeeze(-1)  # (batch,)

            nll = nll_batched(
                model, batch, jax.random.split(key_3, num=ns), beta=hp.beta_nll
            )

            lam = 0.1
            return jnp.mean(lam * adv_error + nll)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array,
        optimizer: nnx.Optimizer = None,  # not used, but we keep the same signature as train_step for simplicity
    ) -> jax.Array:
        keys = jax.random.split(key, num=batch[0].shape[0])
        return jnp.mean(nll_batched(model, batch, keys, beta=hp.beta_nll))

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

        key = jax.random.key(hp.train_rng_seed)
        key, subkey = jax.random.split(key)

        batch_variance_tracking = ds_val[
            jax.random.choice(
                subkey, len(ds_val), shape=(hp.batch_size,), replace=False
            )
        ]
        _, subkey = jax.random.split(subkey)

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
