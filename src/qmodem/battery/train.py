from __future__ import annotations

import pathlib
from collections.abc import Iterable
from enum import StrEnum
from typing import Literal, TypeVar

import jax
import optax
from flax import nnx
from pydantic import BaseModel, ConfigDict

from qmodem.data import DataPipeline
from qmodem.tracking import get_run_parameters
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

from .data_processing import dataloader_builders, prepare_data
from .train_steps import (
    StandardStepFactory,
    StandardStepFactoryContext,
    make_nll_steps,
)


class Method(StrEnum):
    HNN = "hnn"
    MCD = "mcd"
    BNN = "bnn"
    QAVI = "qavi"


class _TrainHyperparametersBase(BaseModel):
    """Hyperparameters shared by every training method."""

    model_config = ConfigDict(frozen=True)

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
    dropout_rate: float = 0.1


class TrainHyperparameters(_TrainHyperparametersBase):
    """Hyperparameters for standard supervised training (HNN, MCD, BNN)."""

    method: Literal[Method.HNN, Method.MCD, Method.BNN] = Method.HNN
    learning_rate: float = 1e-2
    scheduler_alpha: float = 0.1


class QAVITrainHyperparameters(_TrainHyperparametersBase):
    """Hyperparameters for adversarial (QAVI) training."""

    method: Literal[Method.QAVI] = Method.QAVI
    pqc_n_qubits: int = 5
    pqc_n_layers: int = 1
    discriminator_hidden_size: int = 64
    discriminator_act_fn: str = "leaky_relu"
    discriminator_init_seed: int = 43
    learning_rate_generator: float = 1e-3
    learning_rate_discriminator: float = 1e-3
    adversarial_loss_weight: float = 0.1


TrainHpT = TypeVar("TrainHpT", bound=_TrainHyperparametersBase)


def load_train_hyperparameters_from_mlflow(
    model_cls: type[TrainHpT], run_id: str, backend_store: str
) -> TrainHpT:
    """Reload and validate a training hyperparameter set from a training run's logged
    MLflow params.

    MLflow always returns logged params as strings; pydantic's lax validation
    coerces them back to the declared int/float/bool types.

    Args:
        model_cls: Concrete hyperparameter model to validate against
            (``TrainHyperparameters`` or ``QAVITrainHyperparameters``).
        run_id: The ID of the MLflow training run to reload parameters from.
        backend_store: SQLAlchemy URI for the MLflow backend store.

    Returns:
        A validated instance of ``model_cls`` populated from the run's logged
        params.

    Raises:
        pydantic.ValidationError: If the logged params do not match the
            schema of ``model_cls``.
    """
    raw_params = get_run_parameters(run_id, backend_store)
    return model_cls.model_validate(raw_params)


def run_training(
    *,
    model: nnx.Module,
    hp: TrainHyperparameters,
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    data_pipeline: DataPipeline,
    step_factory: StandardStepFactory | None = None,
    callbacks: Iterable[Callback] = (),
) -> None:
    data = prepare_data(
        raw_data_dir,
        data_gen_run_id,
        data_pipeline,
    )
    train_dataloader_builder, val_dataloader_builder = dataloader_builders(
        data,
        batch_size=hp.batch_size,
        drop_remainder=hp.drop_remainder,
    )

    # TODO Unify the arguments of the step factories to a single context
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


def run_adversarial_training(
    *,
    model: nnx.Module,
    discriminator: nnx.Module,
    hp: QAVITrainHyperparameters,
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    data_pipeline: DataPipeline,
    generator_batch_fn: TrainStepFn,
    discriminator_batch_fn: TrainStepFn,
    eval_batch_fn: EvalStepFn | None = None,
    callbacks: Iterable[Callback] = (),
) -> None:
    data = prepare_data(
        raw_data_dir,
        data_gen_run_id,
        pipeline=data_pipeline,
    )
    train_dataloader_builder, val_dataloader_builder = dataloader_builders(
        data,
        batch_size=hp.batch_size,
        drop_remainder=hp.drop_remainder,
    )

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
