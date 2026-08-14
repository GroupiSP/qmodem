from __future__ import annotations

import pathlib
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum

import jax
import optax
from flax import nnx

from qmodem.data import DataPipeline
from qmodem.tracking import MLFlowSetup
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


@dataclass
class TrainHyperparameters:
    method: str = "hnn"
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
    learning_rate: float = 1e-2
    scheduler_alpha: float = 0.1
    pqc_n_qubits: int = 5
    pqc_n_layers: int = 1
    discriminator_hidden_size: int = 64
    discriminator_act_fn: str = "leaky_relu"
    discriminator_init_seed: int = 43
    learning_rate_generator: float = 1e-3
    learning_rate_discriminator: float = 1e-3
    adversarial_loss_weight: float = 0.1

    def _set_qavi_attrs_to_none(self) -> None:
        self.pqc_n_qubits = None
        self.pqc_n_layers = None
        self.discriminator_hidden_size = None
        self.discriminator_act_fn = None
        self.discriminator_init_seed = None
        self.learning_rate_generator = None
        self.learning_rate_discriminator = None
        self.adversarial_loss_weight = None

    def _set_non_qavi_attrs_to_none(self) -> None:
        self.learning_rate = None
        self.scheduler_alpha = None

    def __post_init__(self) -> None:
        if self.method == Method.QAVI:
            self._set_non_qavi_attrs_to_none()
        else:
            self._set_qavi_attrs_to_none()


@contextmanager
def _null_context(setup: MLFlowSetup) -> Generator[None, None, None]:
    """A context manager that does nothing.

    This is useful when the MLFlowSetup is defined at a higher level than training, for
    example when doing hyperparameter optimization, where we do not want to create an
    independent MLFlow run/experiment for each training run, but rather spawn nested
    training runs under the same parent run.
    """
    yield


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
    hp: TrainHyperparameters,
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
        data_pipeline=data_pipeline,
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
