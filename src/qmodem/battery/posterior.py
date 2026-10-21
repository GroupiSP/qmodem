"""The posterior stage shared by every method with a posterior (protocol §4-§5).

Every method run with a posterior stage solves the same posterior problem and is
validated and early-stopped by the same criterion. Only the inference method (and,
for MCD, the posterior network) differs. This module owns what is shared:

- the inferred subnetwork (the whole conv layer, ``INFERRED_SUBNETWORK``),
- the prior ``IsotropicGaussianPrior(precision=1.0)``,
- the validation criterion: mean sample CRPS of the posterior predictive over
  ``PREDICTIVE_SAMPLE_COUNT`` draws, under one validation key per method run
  (``validation_metrics``, ``PosteriorKeys.validation``, ADR 0002),
- early stopping on validation CRPS (``posterior_early_stopper``),
- the stage schedule, batch size and dataset size (``PosteriorStageHyperparameters``),
- the per-method plug-in (``PosteriorSpec``) and the stage builder
  (``build_posterior_stage``),
- the cost metrics of table T3 (``PosteriorCost``, ``log_cost``).
"""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
from flax import nnx
from pydantic import BaseModel, ConfigDict, Field
from stagereg.core import (
    CheckpointStore,
    EarlyStopper,
    EventSink,
    ExperimentTracker,
    IntervalCoverage,
    LoaderFactory,
    NegativeLogLikelihood,
    RootMeanSquaredError,
    SampleContinuousRankedProbabilityScore,
    TrainingEvent,
)
from stagereg.jax import (
    InferenceMethod,
    IsotropicGaussianPrior,
    MetricSuite,
    PosteriorPredictive,
    PosteriorPredictivePredictor,
    PosteriorStage,
    PosteriorStageOptions,
)
from stagereg.jax.metrics import reduce_metric_suite

from qmodem.tracking import Stage

from .data_processing import (
    PreparedData,
    ValidationSet,
    make_loader,
    restore_only_loader,
)
from .evaluate import PRIMARY_COVERAGE_LEVEL, coverage_key
from .train import MeanNetworkHyperparameters

type PyTree = Any
type NetworkBuilder = Callable[[MeanNetworkHyperparameters, jax.Array], nnx.Module]
"""Builds a method's posterior network from the backbone sizes and a key."""

POSTERIOR_CHECKPOINT_KEY = "posterior/best"
"""Store key of the posterior stage's best and finalized checkpoint."""

INFERRED_SUBNETWORK = nnx.PathContains("conv")
"""The inferred subnetwork of every method: the conv layer's kernel and bias.

The remaining parameters (the frozen remainder) stay at their trained-mean values.
"""

PRIOR_PRECISION = 1.0
"""Precision of the isotropic Gaussian prior shared by every posterior."""

PREDICTIVE_SAMPLE_COUNT = 100
"""Draws (and predictive samples) per window: validation and evaluation alike."""

VALIDATION_METRIC = "crps"
"""The validation criterion: early-stopping and HPO metric of every posterior."""

VAL_CRPS_BASELINE = "val_crps_baseline"
"""The validation criterion of the warm-started posterior, before epoch 0.

Diagnostic only: it never feeds the early stopper, checkpoint selection or HPO.
"""

COST_WALL_CLOCK = "cost/posterior/wall_clock_seconds"
COST_TRAINABLE_PARAMETERS = "cost/posterior/trainable_parameters"
COST_DRAWS_PER_SECOND = "cost/posterior/draws_per_second"
DRAWS_PER_SECOND_REPEATS = 5
"""Timed calls of the draws-per-second probe; it reports their median."""
COST_SCAFFOLDING_PARAMETERS = "cost/posterior/scaffolding_parameters"
"""Parameters trained only to fit the posterior, not part of it (AVI's discriminator);
logged only by methods that have any."""
COST_EPOCH_SECONDS = "cost/posterior/epoch_seconds"
"""Wall-clock seconds of each posterior epoch (step = epoch)."""
COST_SECONDS_PER_EPOCH = "cost/posterior/seconds_per_epoch"
"""Mean wall-clock seconds per posterior epoch (step 0)."""


class PosteriorStageHyperparameters(BaseModel):
    """Schedule of a posterior stage, the same for every method (protocol §5).

    Attributes:
        max_epochs: Epoch cap (pSGLD: the fixed number of epochs).
        patience: Non-improving validation epochs tolerated before stopping
            (``min_delta = 0``); unused by methods without early stopping.
        batch_size: Training batch size (Adam at a constant learning rate).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_epochs: int = Field(default=300, ge=1)
    patience: int = Field(default=60, ge=0)
    batch_size: int = Field(default=32, ge=1)


@dataclasses.dataclass(frozen=True)
class PosteriorSpec:
    """How one method plugs into the shared posterior stage.

    Attributes:
        stage: The stage schedule and batch size.
        build_method: Builds the inference method from the posterior-stage
            initialization key (``PosteriorKeys.init``). It is called once per
            training or restore, so it must return a fresh method every time.
        trainable_parameters: Returns the posterior's trained parameters (e.g.
            BBB's variational means and rhos, AVI's generator), whose size is the
            trainable parameter count of cost table T3. Called on the finalized
            method.
        early_stopping: Whether the stage early-stops on validation CRPS
            (``False`` for pSGLD, which finalizes its last epoch).
        build_network: Builds the posterior network (e.g. the MCD dropout net).
            ``None`` uses a clone of the backbone's mean network.
        scaffolding_parameters: Returns the parameters trained only to fit the
            posterior and discarded afterwards (AVI's discriminator), counted
            separately in T3. ``None`` if the method has none.
        method_metrics: Fixed, derived descriptors of the method (e.g. the AVI
            latent maps' parameter counts), logged once as metrics at step 0.
            They are not hyperparameters, so they never enter ``method_config``.
    """

    stage: PosteriorStageHyperparameters
    build_method: Callable[[jax.Array], InferenceMethod]
    trainable_parameters: Callable[[InferenceMethod], PyTree]
    early_stopping: bool = True
    build_network: NetworkBuilder | None = None
    scaffolding_parameters: Callable[[InferenceMethod], PyTree] | None = None
    method_metrics: Mapping[str, float] = dataclasses.field(default_factory=dict)


# --- keys ------------------------------------------------------------------------

_INIT_STREAM = 0x696E6974  # "init"
_NETWORK_STREAM = 0x6E657477  # "netw"
_LOADER_STREAM = 0x6C6F6164  # "load"
_VALIDATION_STREAM = 0x76616C64  # "vald"


@dataclasses.dataclass(frozen=True)
class PosteriorKeys:
    """The keys of a posterior stage, all derived from the method run's key.

    The training state itself is keyed with the method run's key, so update and
    per-epoch keys come from ``state.rng_state``.

    Attributes:
        init: Initializes the inference method (e.g. an AVI generator).
        network: Initializes the posterior network (its Params are then
            overwritten by the mean's; only non-Param state keeps it).
        loader: Shuffles the training windows.
        validation: The fixed key of every validation (stagereg's
            ``PosteriorStageOptions.validation_key``), see ADR 0002.
    """

    init: jax.Array
    network: jax.Array
    loader: jax.Array
    validation: jax.Array

    @classmethod
    def of(cls, method_key: jax.Array) -> PosteriorKeys:
        """Derive the posterior-stage keys of a method run.

        Args:
            method_key: The method run's key ``fold_in(key(r), method)``.

        Returns:
            The posterior-stage keys.
        """
        return cls(
            init=jax.random.fold_in(method_key, _INIT_STREAM),
            network=jax.random.fold_in(method_key, _NETWORK_STREAM),
            loader=jax.random.fold_in(method_key, _LOADER_STREAM),
            validation=jax.random.fold_in(method_key, _VALIDATION_STREAM),
        )


# --- validation ------------------------------------------------------------------


def validation_metrics() -> MetricSuite:
    """Return the shared validation metric suite of every posterior stage.

    CRPS is the criterion; NLL, RMSE and the 95% coverage are diagnostics. Each
    epoch scores ``PREDICTIVE_SAMPLE_COUNT`` mixture samples of the posterior
    predictive; stagereg keys them per batch from the stage's validation key.

    Returns:
        The metric suite.
    """
    return MetricSuite(
        epoch=(
            NegativeLogLikelihood(),
            SampleContinuousRankedProbabilityScore(name=VALIDATION_METRIC),
            RootMeanSquaredError(),
            IntervalCoverage(
                level=PRIMARY_COVERAGE_LEVEL, name=coverage_key(PRIMARY_COVERAGE_LEVEL)
            ),
        ),
        predictor=PosteriorPredictivePredictor(),
        predictive_sample_count=PREDICTIVE_SAMPLE_COUNT,
    )


def validation_crps(
    predictive: Callable[[PyTree, jax.Array], PosteriorPredictive],
    loader: LoaderFactory,
    validation_key: jax.Array,
) -> float:
    """Score the validation criterion of a posterior predictive once.

    The score is the one every posterior epoch reports: it derives its keys from
    ``validation_key`` exactly as stagereg's posterior validation does (``split``
    into a draw key and a metric key, then ``fold_in(metric_key, batch_index)``),
    and scores ``validation_metrics()`` over the loader's ``"validation"`` batches.

    Args:
        predictive: Maps validation inputs and the draw key to the posterior
            predictive with ``PREDICTIVE_SAMPLE_COUNT`` draws.
        loader: The posterior stage's loader.
        validation_key: The stage's validation key (``PosteriorKeys.validation``).

    Returns:
        The validation CRPS.

    Raises:
        RuntimeError: If the validation metric suite has no predictor.
    """
    suite = validation_metrics()
    predictor = suite.predictor
    if predictor is None:
        raise RuntimeError("The validation metric suite has no predictor.")
    draw_key, metric_key = jax.random.split(validation_key)
    parts = [
        predictor(
            lambda inputs: predictive(inputs, draw_key),
            batch,
            suite.prediction_requirements,
            jax.random.fold_in(metric_key, index),
        )
        for index, batch in enumerate(loader(split="validation", epoch=0))
    ]
    metrics = reduce_metric_suite(
        suite=suite, losses=None, batch_metric_values={}, epoch_metric_parts=parts
    )
    return metrics[VALIDATION_METRIC]


def posterior_loader(
    data: PreparedData, batch_size: int, key: jax.Array
) -> LoaderFactory:
    """Return the posterior stage's loader: it validates on the posterior set.

    The validation criterion (early stopping, the baseline, posterior HPO) is
    scored on the posterior validation set, disjoint from the backbone's.

    Args:
        data: The prepared windows.
        batch_size: Training batch size.
        key: Typed PRNG key of the training shuffle.

    Returns:
        The loader (see ``make_loader``).
    """
    return make_loader(
        data, batch_size=batch_size, key=key, validation=ValidationSet.POSTERIOR
    )


def posterior_early_stopper(stage: PosteriorStageHyperparameters) -> EarlyStopper:
    """Return the shared early stopper: validation CRPS, ``min_delta = 0``.

    Args:
        stage: The stage schedule.

    Returns:
        The early stopper.
    """
    return EarlyStopper(
        metric=VALIDATION_METRIC, mode="min", patience=stage.patience, min_delta=0.0
    )


def build_posterior_stage(
    *,
    spec: PosteriorSpec,
    method: InferenceMethod,
    network: MeanNetworkHyperparameters,
    keys: PosteriorKeys,
    loader: LoaderFactory | None = None,
    dataset_size: int | None = None,
    event_sinks: Sequence[EventSink] = (),
    store: CheckpointStore | None = None,
) -> PosteriorStage:
    """Build the shared posterior stage around one method's inference method.

    Without a loader the stage can only be restored (``restore_method_run``).

    Args:
        spec: The method's posterior specification.
        method: The method's inference method (built by ``spec.build_method``).
        network: The backbone sizes (for ``spec.build_network``).
        keys: The posterior-stage keys.
        loader: Train and validation windows; ``None`` for a restore.
        dataset_size: The posterior's dataset size N; ignored for a restore.
        event_sinks: Receive the stage's training events.
        store: Receives the ``posterior/best`` checkpoints.

    Returns:
        The posterior stage.
    """
    training = loader is not None
    return PosteriorStage(
        inference_method=method,
        train_loader=loader if training else restore_only_loader,
        dataset_size=dataset_size if training and dataset_size is not None else 1,
        options=PosteriorStageOptions(
            epochs=spec.stage.max_epochs,
            num_draws=PREDICTIVE_SAMPLE_COUNT,
            validation_loader=loader,
            validation_metrics=validation_metrics(),
            validation_key=keys.validation,
            early_stopper=(
                posterior_early_stopper(spec.stage) if spec.early_stopping else None
            ),
            event_sinks=event_sinks,
            checkpoint_store=store,
            checkpoint_key=POSTERIOR_CHECKPOINT_KEY,
        ),
        prior=IsotropicGaussianPrior(precision=PRIOR_PRECISION),
        model=(
            None
            if spec.build_network is None
            else spec.build_network(network, keys.network)
        ),
        inferred_subnetwork=INFERRED_SUBNETWORK,
    )


# --- cost (table T3) ---------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PosteriorCost:
    """The cost of a method run's posterior stage (table T3).

    Attributes:
        wall_clock_seconds: Wall-clock time of the posterior stage (prepare,
            train, finalize); zero without a posterior stage.
        trainable_parameters: Number of parameters of the posterior the method
            trains.
        draws_per_second: Posterior draws per second at the validation windows.
        scaffolding_parameters: Number of parameters trained only to fit the
            posterior (AVI's discriminator), or ``None`` if there are none.
    """

    wall_clock_seconds: float
    trainable_parameters: int
    draws_per_second: float
    scaffolding_parameters: int | None = None


def parameter_count(tree: PyTree) -> int:
    """Count the scalar entries of a parameter tree.

    Args:
        tree: A pytree of arrays.

    Returns:
        The total number of entries.
    """
    return sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(tree))


def draws_per_second(
    draw: Callable[[], jax.Array],
    num_draws: int,
    repeats: int = DRAWS_PER_SECOND_REPEATS,
    clock: Callable[[], float] = time.perf_counter,
) -> float:
    """Measure the compiled throughput of a draw function.

    ``draw`` is first called eagerly, which materializes any posterior state
    cached on first use (pSGLD's retained samples) outside a trace. It is then
    compiled with ``jax.jit``, called once to compile and timed over ``repeats``
    calls. The median call time is robust to the large spread of single timed
    calls (#135).

    Args:
        draw: Takes ``num_draws`` draws and returns them. Must be jittable.
        num_draws: The number of draws one call takes.
        repeats: The number of timed calls, positive.
        clock: Returns the current time in seconds.

    Returns:
        Draws per second of the median timed call.

    Raises:
        ValueError: If ``repeats`` is not positive.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    jax.block_until_ready(draw())  # materialize cached state
    compiled = jax.jit(draw)
    jax.block_until_ready(compiled())  # compile
    seconds = []
    for _ in range(repeats):
        start = clock()
        jax.block_until_ready(compiled())
        seconds.append(clock() - start)
    return num_draws / max(statistics.median(seconds), 1e-9)


@dataclasses.dataclass
class EpochTimer:
    """An event sink that times every posterior epoch and logs it.

    An epoch's time runs from the previous epoch's end (or ``start``) to its own
    ``epoch_end`` event, so it includes the previous epoch's validation.

    Attributes:
        tracker: Receives ``COST_EPOCH_SECONDS`` at step = epoch.
        stage: The timed stage's name.
        seconds: The measured epoch times, in order.
    """

    tracker: ExperimentTracker
    stage: str = Stage.POSTERIOR
    seconds: list[float] = dataclasses.field(default_factory=list)
    _last: float | None = dataclasses.field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start the clock of the first epoch."""
        self._last = time.perf_counter()

    def on_event(self, event: TrainingEvent) -> None:
        """Time and log an epoch at its ``epoch_end`` event.

        Args:
            event: A training event; others are ignored.

        Raises:
            ValueError: If the timer was not started.
        """
        if event.name != "epoch_end" or event.stage != self.stage:
            return
        if self._last is None:
            raise ValueError("call start before training.")
        now = time.perf_counter()
        self.seconds.append(now - self._last)
        self._last = now
        self.tracker.log_metrics(
            {COST_EPOCH_SECONDS: self.seconds[-1]}, step=event.step
        )

    def log_mean(self) -> None:
        """Log the mean epoch time as ``COST_SECONDS_PER_EPOCH`` at step 0."""
        if self.seconds:
            self.tracker.log_metrics(
                {COST_SECONDS_PER_EPOCH: sum(self.seconds) / len(self.seconds)},
                step=0,
            )


def log_cost(tracker: ExperimentTracker, cost: PosteriorCost) -> None:
    """Log the cost metrics of a method run at step 0.

    Args:
        tracker: The method run's tracker.
        cost: The cost to log.
    """
    metrics = {
        COST_WALL_CLOCK: cost.wall_clock_seconds,
        COST_TRAINABLE_PARAMETERS: float(cost.trainable_parameters),
        COST_DRAWS_PER_SECOND: cost.draws_per_second,
    }
    if cost.scaffolding_parameters is not None:
        metrics[COST_SCAFFOLDING_PARAMETERS] = float(cost.scaffolding_parameters)
    tracker.log_metrics(metrics, step=0)
