from __future__ import annotations

import dataclasses
import functools
import io
import json
import pathlib
from collections.abc import Callable, Iterable

import flax.nnx as nnx
import grain
import jax
import mlflow
import optax
import pandas as pd
import sklearn.preprocessing as skpp

from qmodem.data import (
    DataFrameSource,
    DataPipeline,
    DataSource,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    normalize_ruls,
    to_jax,
)
from qmodem.tracking import MLFlowSetup, track_mlflow
from qmodem.train import (
    LogReporter,
    mlflow_track_losses,
    train_loop,
)
from qmodem.train_adversarial import (
    EvalStepFn,
    TrainStepFn,
)
from qmodem.train_adversarial import (
    LogReporter as AdversarialLogReporter,
)
from qmodem.train_adversarial import (
    mlflow_track_losses as mlflow_track_adversarial_losses,
)
from qmodem.train_adversarial import (
    train_loop as adversarial_train_loop,
)
from qmodem.train_base import Callback, EarlyStopper, mlflow_track_model_best_state
from qmodem.utils import LAST_TRAIN_SETUP_PATH, count_parameters

from .train_steps import (
    StandardStepFactory,
    StandardStepFactoryContext,
    make_nll_steps,
)


@dataclasses.dataclass
class BaseTrainHyperparameters:
    conv_kernel_size: int = 5
    conv_n_filters: int = 4
    batch_size: int = 32
    window_size: int = 20
    stride: int = 1
    normalize_rul: bool = True
    net_init_seed: int = 0
    train_rng_seed: int = 1
    drop_remainder: bool = False
    n_epochs: int = 500
    beta_nll: float = 0.5
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    n_samples_predictive_mean_variance: int = 100
    activation_function: str = "gelu"


@dataclasses.dataclass
class TrainHyperparameters(BaseTrainHyperparameters):
    learning_rate: float = 1e-2
    scheduler_alpha: float = 0.1


@dataclasses.dataclass
class MCDTrainHyperparameters(TrainHyperparameters):
    dropout_rate: float = 0.1


@dataclasses.dataclass
class QAVITrainHyperparameters(BaseTrainHyperparameters):
    pqc_n_qubits: int = 5
    pqc_n_layers: int = 1
    discriminator_hidden_size: int = 64
    discriminator_act_fn: str = "leaky_relu"
    discriminator_init_seed: int = 43
    learning_rate_generator: float = 1e-3
    learning_rate_discriminator: float = 1e-3
    adversarial_loss_weight: float = 0.1


@dataclasses.dataclass
class PreparedData:
    train: DataFrameSource
    val: DataFrameSource
    scaler: skpp.MinMaxScaler


def _split_train_val(
    train_path: pathlib.Path, n_histories_train: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(train_path)
    return (
        data[data["run_id"] < n_histories_train],
        data[data["run_id"] >= n_histories_train],
    )


def _prepare_data(
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    hp: BaseTrainHyperparameters,
) -> PreparedData:
    data_gen_run = mlflow.get_run(data_gen_run_id)
    train_df, val_df = _split_train_val(
        raw_data_dir / "train.csv",
        n_histories_train=int(data_gen_run.data.params["n_histories_train"]),
    )

    scaler = skpp.MinMaxScaler(feature_range=(0, 1))
    common_steps = [
        functools.partial(
            get_time_windows_and_join,
            window_size=hp.window_size,
            stride=hp.stride,
        ),
        add_feature_dimension_to_y,
    ]
    train_pipeline = DataPipeline(
        [
            *common_steps,
            functools.partial(normalize_ruls, transform_fn=scaler.fit_transform)
            if hp.normalize_rul
            else lambda data: data,
            to_jax,
        ]
    )
    val_pipeline = DataPipeline(
        [
            *common_steps,
            functools.partial(normalize_ruls, transform_fn=scaler.transform)
            if hp.normalize_rul
            else lambda data: data,
            to_jax,
        ]
    )

    return PreparedData(
        train=DataFrameSource(df=train_df, pipeline=train_pipeline),
        val=DataFrameSource(df=val_df, pipeline=val_pipeline),
        scaler=scaler,
    )


def _train_dataloader_builder(
    sampler_seed: int,
    ds_train: DataSource,
    batch_size: int,
    drop_remainder: bool,
) -> grain.DataLoader:
    sampler = grain.samplers.IndexSampler(
        num_records=len(ds_train),
        num_epochs=1,
        shuffle=True,
        seed=sampler_seed,
    )
    return grain.DataLoader(
        data_source=ds_train,
        sampler=sampler,
        operations=[
            grain.transforms.Batch(
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        ],
        worker_count=0,
    )


def _dataloader_builders(
    data: PreparedData, hp: BaseTrainHyperparameters
) -> tuple[Callable[[int], Iterable], Callable[[int], Iterable]]:
    train_builder = functools.partial(
        _train_dataloader_builder,
        ds_train=data.train,
        batch_size=hp.batch_size,
        drop_remainder=hp.drop_remainder,
    )

    def val_builder(epoch: int) -> list[tuple[jax.Array, jax.Array]]:
        """The validation dataloader is a convention for consistency with the training
        loop.

        It returns a single batch containing the entire validation set, which is assumed
        to be small enough to fit in memory.
        """
        return [(data.val.X, data.val.y)]

    return train_builder, val_builder


def _write_setup_to_file(experiment_name: str) -> None:
    """Writes only the setup information necessary to resume a trained run for testing
    purposes."""
    active_run = mlflow.active_run()

    with open(LAST_TRAIN_SETUP_PATH, "w") as f:
        json.dump(
            {
                "run_id": active_run.info.run_id,
                "experiment_name": experiment_name,
            },
            f,
        )


def run_training(
    *,
    model: nnx.Module,
    hp: TrainHyperparameters,
    mlflow_setup: MLFlowSetup,
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    log_stream: io.StringIO,
    step_factory: StandardStepFactory | None = None,
    callbacks: Iterable[Callback] = (),
) -> None:
    data = _prepare_data(raw_data_dir, data_gen_run_id, hp)
    train_dataloader_builder, val_dataloader_builder = _dataloader_builders(data, hp)

    if step_factory is None:
        train_batch_fn, eval_batch_fn = make_nll_steps(beta=hp.beta_nll)
    else:
        train_batch_fn, eval_batch_fn = step_factory(
            StandardStepFactoryContext(n_train_samples=len(data.train))
        )

    schedule = optax.cosine_decay_schedule(
        init_value=hp.learning_rate,
        decay_steps=hp.n_epochs * (len(data.train) // hp.batch_size),
        alpha=hp.scheduler_alpha,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)
    early_stopper = EarlyStopper(
        patience=hp.early_stopping_patience,
        min_delta=hp.early_stopping_min_delta,
    )

    with track_mlflow(setup=mlflow_setup):
        _write_setup_to_file(mlflow_setup.experiment_name)

        mlflow.sklearn.log_model(data.scaler, artifact_path="sklearn_scaler")
        mlflow.log_params(dataclasses.asdict(hp))
        mlflow.log_param("n_params", count_parameters(model))

        train_loop(
            n_epochs=hp.n_epochs,
            train_dataloader_builder=train_dataloader_builder,
            val_dataloader_builder=val_dataloader_builder,
            initial_key=jax.random.key(hp.train_rng_seed),
            model=model,
            optimizer=optimizer,
            train_batch_fn=train_batch_fn,
            eval_batch_fn=eval_batch_fn,
            callbacks=(
                LogReporter(log_every=10),
                mlflow_track_model_best_state,
                mlflow_track_losses,
                *callbacks,
            ),
            early_stopper=early_stopper,
        )

        mlflow.log_text(log_stream.getvalue(), "training_log.txt")


def run_adversarial_training(
    *,
    model: nnx.Module,
    discriminator: nnx.Module,
    hp: QAVITrainHyperparameters,
    mlflow_setup: MLFlowSetup,
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    log_stream: io.StringIO,
    generator_batch_fn: TrainStepFn,
    discriminator_batch_fn: TrainStepFn,
    eval_batch_fn: EvalStepFn | None = None,
    callbacks: Iterable[Callback] = (),
) -> None:
    data = _prepare_data(raw_data_dir, data_gen_run_id, hp)
    train_dataloader_builder, val_dataloader_builder = _dataloader_builders(data, hp)

    if eval_batch_fn is None:
        _, eval_batch_fn = make_nll_steps(beta=hp.beta_nll)

    optimizer_generator = nnx.Optimizer(
        model, optax.adam(hp.learning_rate_generator), wrt=nnx.Param
    )
    optimizer_discriminator = nnx.Optimizer(
        discriminator,
        optax.adam(hp.learning_rate_discriminator),
        wrt=nnx.Param,
    )
    early_stopper = EarlyStopper(
        patience=hp.early_stopping_patience,
        min_delta=hp.early_stopping_min_delta,
    )

    with track_mlflow(setup=mlflow_setup):
        _write_setup_to_file(mlflow_setup.experiment_name)

        mlflow.sklearn.log_model(data.scaler, artifact_path="sklearn_scaler")
        mlflow.log_params(dataclasses.asdict(hp))
        mlflow.log_param("n_params", count_parameters(model))

        adversarial_train_loop(
            n_epochs=hp.n_epochs,
            train_dataloader_builder=train_dataloader_builder,
            val_dataloader_builder=val_dataloader_builder,
            initial_key=jax.random.key(hp.train_rng_seed),
            model=model,
            discriminator=discriminator,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            generator_batch_fn=generator_batch_fn,
            discriminator_batch_fn=discriminator_batch_fn,
            eval_batch_fn=eval_batch_fn,
            callbacks=(
                AdversarialLogReporter(log_every=10),
                mlflow_track_model_best_state,
                mlflow_track_adversarial_losses,
                *callbacks,
            ),
            early_stopper=early_stopper,
        )

        mlflow.log_text(log_stream.getvalue(), "training_log.txt")
