from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Literal

import jax
import optax
from flax import nnx
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, ConfigDict, Field
from stagereg.core import (
    Checkpoint,
    CheckpointStore,
    EarlyStopper,
    EventSink,
    ExperimentTracker,
    LoaderFactory,
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

from qmodem.tracking import (
    MLFlowSetup,
    MLflowTracker,
    ScalerArtifact,
    Stage,
    backbone_tags,
    log_scaler,
    mlflow_checkpoint_store,
    track_mlflow,
    unflatten_parameters,
)

from .data_processing import PreparedData, ValidationSet, make_loader
from .models import MeanNet, VarianceNet


class Method(StrEnum):
    """The inference methods of the comparison protocol.

    Values are the ``method`` tag of a method run (glossary names in lowercase).
    """

    HNN = "hnn"
    MCD = "mcd"
    BBB = "bbb"
    PSGLD = "psgld"
    Q_AVI = "q-avi"
    C_AVI = "c-avi"


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

LR_DECAY_FLOOR = 0.01
"""Final over initial learning rate of the mean and variance stages' decay."""


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
    """Adam's learning rate.

    The posterior stage uses it as a constant; the mean and variance stages use
    it as the start of their cosine decay (see ``backbone_optimizer``).

    Attributes:
        learning_rate: Adam's (initial) learning rate.
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


@dataclasses.dataclass(frozen=True)
class EpochCosineDecay:
    """Cosine-decayed learning rate, constant within an epoch (an optax schedule).

    A value-comparable callable (unlike a closure or ``functools.partial``), so
    a checkpointed optimizer compares equal to its reloaded copy.

    Attributes:
        learning_rate: The learning rate at epoch 0.
        max_epochs: The epoch at which the floor is reached (and kept after).
        steps_per_epoch: Optimizer steps in one epoch.
    """

    learning_rate: float
    max_epochs: int
    steps_per_epoch: int

    def __call__(self, step: jax.Array) -> jax.Array:
        """Return the learning rate of ``step``'s epoch.

        Args:
            step: The optimizer step count.

        Returns:
            ``LR_DECAY_FLOOR``-floored cosine decay at epoch
            ``step // steps_per_epoch``.
        """
        decay = optax.cosine_decay_schedule(
            self.learning_rate, decay_steps=self.max_epochs, alpha=LR_DECAY_FLOOR
        )
        return decay(step // self.steps_per_epoch)


def backbone_optimizer(
    optimizer: OptimizerHyperparameters, *, max_epochs: int, steps_per_epoch: int
) -> optax.GradientTransformation:
    """Adam with the mean and variance stages' per-epoch cosine LR decay.

    The learning rate decays from ``optimizer.learning_rate`` at epoch 0 to
    ``LR_DECAY_FLOOR · learning_rate`` at epoch ``max_epochs``, and is stepped
    once per epoch. The floor ratio is fixed, not a hyperparameter.

    Args:
        optimizer: The stage's optimizer hyperparameters.
        max_epochs: The stage's epoch cap.
        steps_per_epoch: Optimizer steps (training batches) per epoch.

    Returns:
        The stage's gradient transformation.

    Raises:
        ValueError: If ``max_epochs`` or ``steps_per_epoch`` is below 1.
    """
    if max_epochs < 1 or steps_per_epoch < 1:
        raise ValueError(
            "max_epochs and steps_per_epoch must be >= 1, got "
            f"{max_epochs} and {steps_per_epoch}."
        )
    return optax.adam(
        EpochCosineDecay(
            learning_rate=optimizer.learning_rate,
            max_epochs=max_epochs,
            steps_per_epoch=steps_per_epoch,
        )
    )


def steps_per_epoch(data: PreparedData, batch_size: int) -> int:
    """Optimizer steps per epoch: training batches, the last one possibly smaller.

    Args:
        data: The prepared windows.
        batch_size: Training batch size.

    Returns:
        ``ceil(n_train_windows / batch_size)``, at least 1.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    return max(1, math.ceil(len(data.train.features) / batch_size))


def _early_stopper(schedule: StageScheduleHyperparameters) -> EarlyStopper:
    return EarlyStopper(
        metric="loss", mode="min", patience=schedule.patience, min_delta=0.0
    )


def _mean_stage(
    *,
    mean_hp: MeanHyperparameters,
    keys: BackboneKeys,
    loader: LoaderFactory,
    steps_per_epoch: int,
    event_sinks: Sequence[EventSink],
    store: CheckpointStore,
) -> MeanStage:
    """Build the mean stage of a backbone run.

    Args:
        mean_hp: Mean-stage hyperparameters.
        keys: The backbone run's keys.
        loader: The backbone run's loader.
        steps_per_epoch: Training batches per epoch (the LR decay's epoch length).
        event_sinks: Receive the stage's training events.
        store: Receives the ``mean/best`` checkpoints.

    Returns:
        The mean stage, on a freshly initialized mean network.
    """
    mean_net = build_mean_net(mean_hp.network, keys.mean_init)
    mean_loss = make_supervised_loss(SquaredErrorLoss())
    return MeanStage(
        model=mean_net,
        optimizer=create_optimizer(
            mean_net,
            backbone_optimizer(
                mean_hp.optimizer,
                max_epochs=mean_hp.schedule.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
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


def _variance_stage(
    *,
    network: MeanNetworkHyperparameters,
    variance_hp: VarianceHyperparameters,
    keys: BackboneKeys,
    loader: LoaderFactory,
    steps_per_epoch: int,
    event_sinks: Sequence[EventSink],
    store: CheckpointStore,
) -> GammaVarianceStage:
    """Build the variance stage of a backbone run.

    Args:
        network: Backbone sizes, taken from the mean network's configuration.
        variance_hp: Variance-stage hyperparameters.
        keys: The backbone run's keys.
        loader: The backbone run's loader (the residuals' source).
        steps_per_epoch: Training batches per epoch (the LR decay's epoch length).
        event_sinks: Receive the stage's training events.
        store: Receives the ``variance/best`` checkpoints.

    Returns:
        The variance stage, on a freshly initialized variance network.
    """
    variance_net = build_variance_net(network, keys.variance_init)
    variance_stage = GammaVarianceStage(
        model=variance_net,
        optimizer=create_optimizer(
            variance_net,
            backbone_optimizer(
                variance_hp.optimizer,
                max_epochs=variance_hp.schedule.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
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
    return variance_stage


def train_mean_stage(
    *,
    data: PreparedData,
    mean_hp: MeanHyperparameters,
    seed: int,
    tracker: ExperimentTracker,
    store: CheckpointStore,
    event_sinks: Sequence[EventSink] = (),
) -> Checkpoint:
    """Train the mean stage of a backbone run on its own (a mean-study trial).

    It is the first half of ``train_backbone``: the same seed and hyperparameters
    give the backbone run's ``mean/best`` checkpoint bit for bit.

    Args:
        data: Prepared train and validation windows.
        mean_hp: Mean-stage hyperparameters.
        seed: The protocol seed r.
        tracker: Receives the mean hyperparameters and the stage metrics.
        store: Receives the ``mean/best`` checkpoints.
        event_sinks: Extra sinks of the stage's training events, e.g. an HPO
            pruner.

    Returns:
        The finalized ``mean/best`` checkpoint.
    """
    tracker.log_params({Stage.MEAN: mean_hp.model_dump()})
    keys = backbone_keys(seed)
    stage = _mean_stage(
        mean_hp=mean_hp,
        keys=keys,
        loader=make_loader(
            data,
            batch_size=mean_hp.batch_size,
            key=keys.loader,
            validation=ValidationSet.BACKBONE,
        ),
        steps_per_epoch=steps_per_epoch(data, mean_hp.batch_size),
        event_sinks=(TrackerEventSink(tracker), *event_sinks),
        store=store,
    )
    state = TrainingState(rng_state=keys.training)
    stage.prepare(state)
    stage.train(state)
    return store.load(MEAN_CHECKPOINT_KEY)


def train_variance_stage(
    *,
    data: PreparedData,
    mean_hp: MeanHyperparameters,
    mean_checkpoint: Checkpoint,
    variance_hp: VarianceHyperparameters,
    seed: int,
    tracker: ExperimentTracker,
    store: CheckpointStore,
    event_sinks: Sequence[EventSink] = (),
) -> Checkpoint:
    """Train the variance stage on a frozen mean checkpoint (a variance-study trial).

    It is the second half of ``train_backbone``: on the backbone run's ``mean/best``
    checkpoint, the same seed and hyperparameters give its ``variance/best``
    checkpoint bit for bit.

    Args:
        data: Prepared train and validation windows.
        mean_hp: The mean-stage hyperparameters ``mean_checkpoint`` was trained
            with (they size both networks and fix the batch size).
        mean_checkpoint: The finalized ``mean/best`` checkpoint to restore.
        variance_hp: Variance-stage hyperparameters.
        seed: The protocol seed r of the mean checkpoint.
        tracker: Receives the variance hyperparameters and the stage metrics.
        store: Receives the ``variance/best`` checkpoints.
        event_sinks: Extra sinks of the stage's training events, e.g. an HPO
            pruner.

    Returns:
        The finalized ``variance/best`` checkpoint.
    """
    tracker.log_params({Stage.VARIANCE: variance_hp.model_dump()})
    keys = backbone_keys(seed)
    loader = make_loader(
        data,
        batch_size=mean_hp.batch_size,
        key=keys.loader,
        validation=ValidationSet.BACKBONE,
    )
    n_steps = steps_per_epoch(data, mean_hp.batch_size)
    state = TrainingState()
    mean_stage = _mean_stage(
        mean_hp=mean_hp,
        keys=keys,
        loader=loader,
        steps_per_epoch=n_steps,
        event_sinks=(),
        store=store,
    )
    mean_stage.restore(state, mean_checkpoint)
    stage = _variance_stage(
        network=mean_hp.network,
        variance_hp=variance_hp,
        keys=keys,
        loader=loader,
        steps_per_epoch=n_steps,
        event_sinks=(TrackerEventSink(tracker), *event_sinks),
        store=store,
    )
    stage.prepare(state)
    stage.train(state)
    return store.load(VARIANCE_CHECKPOINT_KEY)


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
    ``seed``, so the same seed gives bit-identical checkpoints. It is
    ``train_mean_stage`` followed by ``train_variance_stage`` on its result, the
    same operations the mean and variance HPO trials run.

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
    mean = train_mean_stage(
        data=data, mean_hp=mean_hp, seed=seed, tracker=tracker, store=store
    )
    variance = train_variance_stage(
        data=data,
        mean_hp=mean_hp,
        mean_checkpoint=mean,
        variance_hp=variance_hp,
        seed=seed,
        tracker=tracker,
        store=store,
    )
    return BackboneCheckpoints(mean=mean, variance=variance)


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
    mean_study_run_id: str | None = None,
    variance_study_run_id: str | None = None,
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
        mean_study_run_id: The mean study the mean HPs come from, if any
            (recorded as a tag).
        variance_study_run_id: The variance study the variance LR comes from, if
            any (recorded as a tag; posterior studies require it).

    Returns:
        The run ID and the finalized checkpoints.
    """
    tags = backbone_tags(
        seed=seed,
        data_generation_run_id=data_generation_run_id,
        mean_study_run_id=mean_study_run_id,
        variance_study_run_id=variance_study_run_id,
    )
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


def load_backbone_hyperparameters(
    run_id: str, client: MlflowClient | None = None
) -> tuple[MeanHyperparameters, VarianceHyperparameters]:
    """Reload the mean and variance hyperparameters a backbone run logged.

    Args:
        run_id: The backbone run.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        The validated mean and variance hyperparameters.

    Raises:
        pydantic.ValidationError: If the logged params do not match the schemas.
    """
    client = client if client is not None else MlflowClient()
    nested = unflatten_parameters(client.get_run(run_id).data.params)
    return (
        MeanHyperparameters.model_validate(nested[Stage.MEAN]),
        VarianceHyperparameters.model_validate(nested[Stage.VARIANCE]),
    )
