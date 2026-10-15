"""Method runs: restore a backbone run and add one inference method's posterior stage.

A method is described by a small per-method configuration (``MethodConfig``), and
``METHOD_CONFIGS`` maps every ``Method`` to its configuration type (strategy
pattern), so adding a method needs no change to the pipeline. The pipeline itself
is ``train_method_run`` (in the run that trains the posterior) and
``restore_method_run`` (in any later process, e.g. evaluation).

Every key of a method run derives from ``method_key(seed, method)``, i.e.
``fold_in(key(r), METHOD_KEY_INDEX[method])``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Protocol

import jax
import jax.numpy as jnp
import optax
from pydantic import BaseModel, ConfigDict
from stagereg.core import (
    Batch,
    Checkpoint,
    CheckpointStore,
    ExperimentTracker,
    TrainingState,
)
from stagereg.jax import (
    GammaVarianceStage,
    MeanStage,
    Posterior,
    PosteriorPredictive,
    SupervisedStageOptions,
    create_optimizer,
)

from .data_processing import PreparedData
from .models import MeanNet, VarianceNet
from .train import (
    MEAN_CHECKPOINT_KEY,
    VARIANCE_CHECKPOINT_KEY,
    MeanHyperparameters,
    Method,
    VarianceHyperparameters,
    backbone_keys,
    build_mean_net,
    build_variance_net,
)

METHOD_KEY_INDEX: Mapping[Method, int] = {
    Method.HNN: 0,
    Method.MCD: 1,
    Method.BBB: 2,
    Method.PSGLD: 3,
    Method.Q_AVI: 4,
    Method.C_AVI: 5,
}
"""Fixed integer of each method, folded into the protocol seed's key.

Never derive it from ``hash()``, which is salted per process.
"""


def method_key(seed: int, method: Method) -> jax.Array:
    """Return the root key of a method run, ``fold_in(key(r), method)``.

    Methods of one seed are thereby paired by seed yet independent.

    Args:
        seed: The protocol seed r.
        method: The inference method.

    Returns:
        The method run's root key.
    """
    return jax.random.fold_in(jax.random.key(seed), METHOD_KEY_INDEX[method])


class MethodConfig(Protocol):
    """The per-method configuration of a method run.

    Each implementation is a frozen data container of the method's posterior
    hyperparameters, registered in ``METHOD_CONFIGS``.
    """

    method: ClassVar[Method]

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the posterior hyperparameters, nested, logged under ``posterior``.

        Returns:
            The hyperparameters (empty for a method without posterior stage).
        """
        ...


class _MethodConfigBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HNNConfig(_MethodConfigBase):
    """HNN: no posterior stage; its predictive is the backbone's mean and variance."""

    method: ClassVar[Method] = Method.HNN

    def posterior_hyperparameters(self) -> dict[str, Any]:
        """Return the (empty) posterior hyperparameters of HNN.

        Returns:
            An empty mapping: HNN has no posterior stage.
        """
        return self.model_dump()


METHOD_CONFIGS: Mapping[Method, type[MethodConfig]] = {
    Method.HNN: HNNConfig,
}
"""The configuration type of every implemented method."""


def method_config(method: Method, hyperparameters: Mapping[str, Any]) -> MethodConfig:
    """Build a method's configuration from its (possibly string-valued) params.

    Args:
        method: The inference method.
        hyperparameters: The posterior hyperparameters, nested as logged by
            ``MethodConfig.posterior_hyperparameters``.

    Returns:
        The validated configuration.

    Raises:
        NotImplementedError: If the method has no registered configuration yet.
        pydantic.ValidationError: If the hyperparameters do not fit the method.
    """
    if method not in METHOD_CONFIGS:
        raise NotImplementedError(f"Method {method.value!r} is not implemented yet.")
    return METHOD_CONFIGS[method].model_validate(dict(hyperparameters))


@dataclasses.dataclass(frozen=True)
class BackboneHandle:
    """What a method run needs to restore its backbone run.

    Attributes:
        mean_hp: The backbone's mean hyperparameters (they size both networks).
        variance_hp: The backbone's variance hyperparameters.
        store: A store holding the finalized ``mean/best`` and ``variance/best``.
    """

    mean_hp: MeanHyperparameters
    variance_hp: VarianceHyperparameters
    store: CheckpointStore


@dataclasses.dataclass(frozen=True)
class MethodRun:
    """A trained or restored method run.

    Attributes:
        method: The inference method.
        key: The method run's root key ``fold_in(key(r), method)``.
        mean_net: The backbone's mean network.
        variance_net: The backbone's variance network.
        posterior: The posterior over the inferred subnetwork, ``None`` for HNN.
        checkpoint: The finalized ``posterior`` checkpoint, ``None`` for HNN.
    """

    method: Method
    key: jax.Array
    mean_net: MeanNet
    variance_net: VarianceNet
    posterior: Posterior | None = None
    checkpoint: Checkpoint | None = None

    def predictive(
        self, inputs: jax.Array, key: jax.Array, num_draws: int
    ) -> PosteriorPredictive:
        """Return the posterior predictive at ``inputs`` over ``num_draws`` draws.

        Without a posterior (HNN), every draw is the backbone's single mean.

        Args:
            inputs: Windows ``(B, W, C)``.
            key: Key of the posterior draws.
            num_draws: Number of draws M.

        Returns:
            The draws' means ``(M, B, 1)`` with the aleatoric variance ``(B, 1)``.
        """
        if self.posterior is None:
            mean = self.mean_net(inputs)
            draws = jnp.broadcast_to(mean, (num_draws, *mean.shape))
        else:
            draws = self.posterior.sample_means(inputs, key, num_draws)
        return PosteriorPredictive(
            draws=draws, aleatoric_variance=self.variance_net(inputs).mean()
        )


def _restore_only_loader(*, split: str, epoch: int) -> Iterable[Batch]:
    """A loader for stages that are only restored, never trained.

    Raises:
        RuntimeError: Always; a restored stage must not read data.
    """
    raise RuntimeError("A restored backbone stage must not read data.")


def _restore_backbone(
    backbone: BackboneHandle, seed: int
) -> tuple[TrainingState, MeanNet, VarianceNet]:
    """Restore a backbone's finalized mean and variance stages into a fresh state.

    Args:
        backbone: The backbone to restore.
        seed: The protocol seed r (only seeds the throwaway initial weights).

    Returns:
        The variance-ready training state and the restored networks.
    """
    keys = backbone_keys(seed)
    network = backbone.mean_hp.network
    state = TrainingState()

    mean_net = build_mean_net(network, keys.mean_init)
    MeanStage(
        model=mean_net,
        optimizer=create_optimizer(
            mean_net, optax.adam(backbone.mean_hp.optimizer.learning_rate)
        ),
        train_loader=_restore_only_loader,
        options=SupervisedStageOptions(epochs=1, checkpoint_key=MEAN_CHECKPOINT_KEY),
    ).restore(state, backbone.store.load(MEAN_CHECKPOINT_KEY))

    variance_net = build_variance_net(network, keys.variance_init)
    GammaVarianceStage(
        model=variance_net,
        optimizer=create_optimizer(
            variance_net, optax.adam(backbone.variance_hp.optimizer.learning_rate)
        ),
        source_loader=_restore_only_loader,
        options=SupervisedStageOptions(
            epochs=1, checkpoint_key=VARIANCE_CHECKPOINT_KEY
        ),
    ).restore(state, backbone.store.load(VARIANCE_CHECKPOINT_KEY))
    return state, mean_net, variance_net


def _require_registered(config: MethodConfig) -> None:
    """Refuse methods whose posterior stage is not wired into the pipeline yet.

    Args:
        config: The method's configuration.

    Raises:
        NotImplementedError: If the method has no registered configuration.
    """
    if config.method not in METHOD_CONFIGS:
        raise NotImplementedError(
            f"Method {config.method.value!r} is not implemented yet."
        )


def train_method_run(
    *,
    backbone: BackboneHandle,
    data: PreparedData,
    config: MethodConfig,
    seed: int,
    tracker: ExperimentTracker,
    store: CheckpointStore,
) -> MethodRun:
    """Train a method run on a restored backbone.

    The backbone's finalized ``mean`` and ``variance`` checkpoints are restored,
    never re-trained. The training state is then re-keyed with
    ``method_key(seed, method)``. HNN trains nothing and saves nothing.

    Args:
        backbone: The backbone run to restore.
        data: Prepared train and validation windows (for the posterior stage).
        config: The method's configuration.
        seed: The protocol seed r.
        tracker: Receives the posterior hyperparameters and stage metrics.
        store: Receives the method run's ``posterior`` checkpoints.

    Returns:
        The method run, with its finalized ``posterior`` checkpoint (``None`` for
        HNN).

    Raises:
        NotImplementedError: If the method is not implemented yet.
    """
    _require_registered(config)
    hyperparameters = config.posterior_hyperparameters()
    if hyperparameters:
        tracker.log_params({"posterior": hyperparameters})
    state, mean_net, variance_net = _restore_backbone(backbone, seed)
    key = method_key(seed, config.method)
    state.rng_state = key
    return MethodRun(
        method=config.method, key=key, mean_net=mean_net, variance_net=variance_net
    )


def restore_method_run(
    *,
    backbone: BackboneHandle,
    config: MethodConfig,
    seed: int,
    store: CheckpointStore,
) -> MethodRun:
    """Restore a trained method run in a fresh process.

    Args:
        backbone: The backbone run the method run was trained on.
        config: The method's configuration.
        seed: The protocol seed r.
        store: The method run's store, holding its finalized ``posterior``
            checkpoint (none for HNN).

    Returns:
        The restored method run.

    Raises:
        NotImplementedError: If the method is not implemented yet.
    """
    _require_registered(config)
    _, mean_net, variance_net = _restore_backbone(backbone, seed)
    return MethodRun(
        method=config.method,
        key=method_key(seed, config.method),
        mean_net=mean_net,
        variance_net=variance_net,
    )
