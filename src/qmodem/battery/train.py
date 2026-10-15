from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterable, Sequence
from enum import StrEnum
from typing import Literal, TypeVar

import jax
import optax
from flax import nnx
from pydantic import BaseModel, ConfigDict, Field
from stagereg.core import (
    Checkpoint,
    CheckpointStore,
    EarlyStopper,
    ExperimentTracker,
    SquaredErrorLoss,
    TrackerEventSink,
    TrainingState,
)
from stagereg.jax import (
    GammaVarianceStage,
    HeldOutValidation,
    MeanStage,
    SupervisedStageOptions,
    create_optimizer,
    make_supervised_loss,
)

from qmodem.data import DataPipeline
from qmodem.tracking import (
    MLFlowSetup,
    MLflowTracker,
    ScalerArtifact,
    backbone_tags,
    get_run_parameters,
    log_scaler,
    mlflow_checkpoint_store,
    track_mlflow,
)
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
from qmodem.train_base import Callback, mlflow_track_model_best_state
from qmodem.train_base import EarlyStopper as LegacyEarlyStopper

from .data_processing import (
    PreparedData,
    dataloader_builders,
    make_loader,
    prepare_data,
)
from .models import MeanNet, VarianceNet
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
    early_stopper = LegacyEarlyStopper(
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
    early_stopper = LegacyEarlyStopper(
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


# --- Backbone run (stagereg) ---------------------------------------------------

MEAN_CHECKPOINT_KEY = "mean/best"
VARIANCE_CHECKPOINT_KEY = "variance/best"

_BACKBONE_NAMESPACE = (
    0x6261636B  # "back": keeps backbone keys apart from fold_in(r, method)
)
_MEAN_INIT_STREAM = 0
_VARIANCE_INIT_STREAM = 1
_LOADER_STREAM = 2
_TRAINING_STREAM = 3


class _FrozenHyperparameters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MeanNetworkHyperparameters(_FrozenHyperparameters):
    """Backbone sizes, shared by the mean and the variance network.

    Attributes:
        kernel_size: Width of the 1D convolution kernel.
        n_filters: Number of convolution filters.
        activation: Name of the ``flax.nnx`` activation after the convolution.
    """

    kernel_size: int = Field(default=3, ge=1)
    n_filters: int = Field(default=29, ge=1)
    activation: Literal["gelu", "relu", "silu", "tanh"] = "gelu"


class OptimizerHyperparameters(_FrozenHyperparameters):
    """Adam at a constant learning rate.

    Attributes:
        learning_rate: Adam's learning rate.
    """

    learning_rate: float = Field(default=1e-3, gt=0.0)


class StageScheduleHyperparameters(_FrozenHyperparameters):
    """Epoch cap and early stopping on the validation loss (``min_delta = 0``).

    Attributes:
        max_epochs: Maximum number of epochs.
        patience: Non-improving epochs tolerated before stopping.
    """

    max_epochs: int = Field(default=500, ge=1)
    patience: int = Field(default=50, ge=0)


class MeanHyperparameters(_FrozenHyperparameters):
    """Hyperparameters of the mean stage.

    Attributes:
        network: Backbone sizes (inherited by the variance network).
        optimizer: Mean-stage optimizer.
        schedule: Mean-stage epochs and early stopping.
        batch_size: Training batch size of the backbone run.
    """

    network: MeanNetworkHyperparameters = MeanNetworkHyperparameters()
    optimizer: OptimizerHyperparameters = OptimizerHyperparameters()
    schedule: StageScheduleHyperparameters = StageScheduleHyperparameters()
    batch_size: int = Field(default=32, ge=1)


class VarianceHyperparameters(_FrozenHyperparameters):
    """Hyperparameters of the variance stage (sizes come from the mean network).

    Attributes:
        optimizer: Variance-stage optimizer.
        schedule: Variance-stage epochs and early stopping.
    """

    optimizer: OptimizerHyperparameters = OptimizerHyperparameters()
    schedule: StageScheduleHyperparameters = StageScheduleHyperparameters()


@dataclasses.dataclass(frozen=True)
class BackboneCheckpoints:
    """The finalized checkpoints of a backbone run.

    Attributes:
        mean: Finalized ``mean/best`` checkpoint.
        variance: Finalized ``variance/best`` checkpoint.
    """

    mean: Checkpoint
    variance: Checkpoint


@dataclasses.dataclass(frozen=True)
class BackboneKeys:
    """Every RNG key of a backbone run, derived from the protocol seed r.

    Attributes:
        mean_init: Initializes the mean network.
        variance_init: Initializes the variance network.
        loader: Shuffles the training windows.
        training: The training state's key.
    """

    mean_init: jax.Array
    variance_init: jax.Array
    loader: jax.Array
    training: jax.Array


def backbone_keys(seed: int) -> BackboneKeys:
    """Derive every key of a backbone run from the protocol seed.

    The keys live under a backbone namespace, so they never coincide with a
    method run's ``fold_in(key(r), method)``.

    Args:
        seed: The protocol seed r.

    Returns:
        The backbone run's keys.
    """
    root = jax.random.fold_in(jax.random.key(seed), _BACKBONE_NAMESPACE)
    return BackboneKeys(
        mean_init=jax.random.fold_in(root, _MEAN_INIT_STREAM),
        variance_init=jax.random.fold_in(root, _VARIANCE_INIT_STREAM),
        loader=jax.random.fold_in(root, _LOADER_STREAM),
        training=jax.random.fold_in(root, _TRAINING_STREAM),
    )


def build_mean_net(network: MeanNetworkHyperparameters, key: jax.Array) -> MeanNet:
    """Build the mean network.

    Args:
        network: Backbone sizes.
        key: Initialization key.

    Returns:
        A freshly initialized mean network.
    """
    return MeanNet(
        kernel_size=network.kernel_size,
        n_filters=network.n_filters,
        activation=getattr(nnx, network.activation),
        rngs=nnx.Rngs(key),
    )


def build_variance_net(
    network: MeanNetworkHyperparameters, key: jax.Array
) -> VarianceNet:
    """Build the variance network with the mean network's sizes.

    Args:
        network: Backbone sizes, taken from the mean network's configuration.
        key: Initialization key.

    Returns:
        A freshly initialized variance network.
    """
    return VarianceNet(
        kernel_size=network.kernel_size,
        n_filters=network.n_filters,
        activation=getattr(nnx, network.activation),
        rngs=nnx.Rngs(key),
    )


def _early_stopper(schedule: StageScheduleHyperparameters) -> EarlyStopper:
    return EarlyStopper(
        metric="loss", mode="min", patience=schedule.patience, min_delta=0.0
    )


def train_backbone(
    *,
    data: PreparedData,
    mean_hp: MeanHyperparameters,
    variance_hp: VarianceHyperparameters,
    seed: int,
    tracker: ExperimentTracker,
    store: CheckpointStore,
) -> BackboneCheckpoints:
    """Train a backbone run: ``MeanStage`` then ``GammaVarianceStage``.

    Both stages early-stop on their validation loss (MSE, then Gamma NLL) and
    finalize their best epoch into ``store``. Every RNG key derives from
    ``seed``, so the same seed gives bit-identical checkpoints.

    Args:
        data: Prepared train and validation windows.
        mean_hp: Mean-stage hyperparameters (they also size the variance net).
        variance_hp: Variance-stage hyperparameters.
        seed: The protocol seed r.
        tracker: Receives the hyperparameters and every stage metric.
        store: Receives the stage checkpoints.

    Returns:
        The finalized ``mean`` and ``variance`` checkpoints.
    """
    tracker.log_params(
        {"mean": mean_hp.model_dump(), "variance": variance_hp.model_dump()}
    )
    keys = backbone_keys(seed)
    loader = make_loader(data, batch_size=mean_hp.batch_size, key=keys.loader)
    event_sinks = (TrackerEventSink(tracker),)
    state = TrainingState(rng_state=keys.training)

    mean_net = build_mean_net(mean_hp.network, keys.mean_init)
    mean_loss = make_supervised_loss(SquaredErrorLoss())
    mean_stage = MeanStage(
        model=mean_net,
        optimizer=create_optimizer(
            mean_net, optax.adam(mean_hp.optimizer.learning_rate)
        ),
        train_loader=loader,
        options=SupervisedStageOptions(
            epochs=mean_hp.schedule.max_epochs,
            validation=HeldOutValidation(model=mean_net, loader=loader, loss=mean_loss),
            early_stopper=_early_stopper(mean_hp.schedule),
            event_sinks=event_sinks,
            checkpoint_store=store,
            checkpoint_key=MEAN_CHECKPOINT_KEY,
        ),
        loss=mean_loss,
    )
    mean_stage.prepare(state)
    mean_stage.train(state)

    variance_net = build_variance_net(mean_hp.network, keys.variance_init)
    variance_stage = GammaVarianceStage(
        model=variance_net,
        optimizer=create_optimizer(
            variance_net, optax.adam(variance_hp.optimizer.learning_rate)
        ),
        source_loader=loader,
        options=SupervisedStageOptions(
            epochs=variance_hp.schedule.max_epochs,
            early_stopper=_early_stopper(variance_hp.schedule),
            event_sinks=event_sinks,
            checkpoint_store=store,
            checkpoint_key=VARIANCE_CHECKPOINT_KEY,
        ),
        validation_factory=lambda residuals: HeldOutValidation(
            model=variance_net, loader=residuals, loss=variance_stage.loss
        ),
    )
    variance_stage.prepare(state)
    variance_stage.train(state)

    return BackboneCheckpoints(
        mean=store.load(MEAN_CHECKPOINT_KEY),
        variance=store.load(VARIANCE_CHECKPOINT_KEY),
    )


@dataclasses.dataclass(frozen=True)
class TrackedBackbone:
    """A backbone run recorded in MLflow.

    Attributes:
        run_id: The backbone run's MLflow run ID.
        checkpoints: Its finalized checkpoints (also logged as run artifacts).
    """

    run_id: str
    checkpoints: BackboneCheckpoints


def track_backbone_run(
    *,
    data: PreparedData,
    mean_hp: MeanHyperparameters,
    variance_hp: VarianceHyperparameters,
    seed: int,
    data_generation_run_id: str,
    scalers: Sequence[ScalerArtifact],
    setup: MLFlowSetup,
) -> TrackedBackbone:
    """Train a backbone run inside an MLflow run.

    The run is tagged with ``backbone_tags``. It receives the mean and variance
    hyperparameters (flattened by stage), every stage metric, the finalized ``mean``
    and ``variance`` checkpoints (``checkpoints/<stage>/best``) and the scalers.

    Args:
        data: Prepared train and validation windows.
        mean_hp: Mean-stage hyperparameters.
        variance_hp: Variance-stage hyperparameters.
        seed: The protocol seed r.
        data_generation_run_id: MLflow run ID of the data-generation run.
        scalers: The scalers fitted on the training histories.
        setup: Tracking backend, experiment and run name. Its tags are extended
            with the backbone tags.

    Returns:
        The run ID and the finalized checkpoints.
    """
    tags = backbone_tags(seed=seed, data_generation_run_id=data_generation_run_id)
    with track_mlflow(dataclasses.replace(setup, tags={**setup.tags, **tags})) as run:
        run_id = run.info.run_id
        checkpoints = train_backbone(
            data=data,
            mean_hp=mean_hp,
            variance_hp=variance_hp,
            seed=seed,
            tracker=MLflowTracker(run_id),
            store=mlflow_checkpoint_store(run_id),
        )
        for scaler in scalers:
            log_scaler(scaler)
    return TrackedBackbone(run_id=run_id, checkpoints=checkpoints)
