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
import numpy as np
import optax
import pandas as pd
from mlflow.tracking import MlflowClient
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

from qmodem.tracking import (
    MLFlowSetup,
    MLflowTracker,
    load_scaler,
    mlflow_checkpoint_store,
    stagereg_version,
    track_mlflow,
)

from .data_processing import PreparedData
from .evaluate import (
    Evaluation,
    EvaluationParameters,
    evaluate_predictive_model,
    log_evaluation,
)
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

_EVALUATION_STREAM = 0x6576616C  # "eval": folded into the method run's root key


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

    @property
    def evaluation_key(self) -> jax.Array:
        """The key of the run's test evaluation, derived from its root key."""
        return jax.random.fold_in(self.key, _EVALUATION_STREAM)

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


# --- MLflow: method runs as children of their backbone run -----------------------

PARENT_RUN_TAG = "mlflow.parentRunId"


def method_tags(*, method: Method, seed: int, backbone_run_id: str) -> dict[str, str]:
    """Return the tags of a method run.

    Args:
        method: The inference method.
        seed: The protocol seed r.
        backbone_run_id: The MLflow run ID of the backbone run (the parent).

    Returns:
        The ``role``, ``method``, ``seed``, ``stagereg_version`` and
        ``mlflow.parentRunId`` tags.
    """
    return {
        "role": "method",
        "method": method.value,
        "seed": str(seed),
        "stagereg_version": stagereg_version(),
        PARENT_RUN_TAG: backbone_run_id,
    }


def _unflatten(params: Mapping[str, str], stage: str) -> dict[str, Any]:
    """Nest the flattened params of one stage (``stage/a/b`` -> ``{a: {b: ..}}``).

    Args:
        params: MLflow run params, flattened by stage.
        stage: The stage prefix, e.g. ``"mean"``.

    Returns:
        The stage's params, nested.
    """
    nested: dict[str, Any] = {}
    for key, value in params.items():
        head, *path = key.split("/")
        if head != stage or not path:
            continue
        node = nested
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = value
    return nested


def load_backbone_handle(
    run_id: str, client: MlflowClient | None = None
) -> BackboneHandle:
    """Return the handle of a backbone run recorded in MLflow.

    Args:
        run_id: The backbone run's ID.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        Its hyperparameters (from the run's params) and its checkpoint store.
    """
    client = client if client is not None else MlflowClient()
    params = client.get_run(run_id).data.params
    return BackboneHandle(
        mean_hp=MeanHyperparameters.model_validate(_unflatten(params, "mean")),
        variance_hp=VarianceHyperparameters.model_validate(
            _unflatten(params, "variance")
        ),
        store=mlflow_checkpoint_store(run_id, client=client),
    )


@dataclasses.dataclass(frozen=True)
class TrackedMethodRun:
    """A method run recorded in MLflow.

    Attributes:
        run_id: The method run's MLflow run ID.
        run: The trained method run.
    """

    run_id: str
    run: MethodRun


def track_method_run(
    *,
    backbone_run_id: str,
    data: PreparedData,
    config: MethodConfig,
    seed: int,
    setup: MLFlowSetup,
    client: MlflowClient | None = None,
) -> TrackedMethodRun:
    """Train a method run inside an MLflow run, as a child of its backbone run.

    The run is tagged with ``method_tags``. It receives the posterior
    hyperparameters (flattened under ``posterior``), the posterior-stage metrics
    and the finalized ``checkpoints/posterior/best`` (none for HNN). The backbone
    is restored from the parent run's checkpoints.

    Args:
        backbone_run_id: The backbone run's ID.
        data: Prepared train and validation windows.
        config: The method's configuration.
        seed: The protocol seed r (should match the backbone's ``seed`` tag).
        setup: Tracking backend, experiment and run name. Its tags are extended
            with the method-run tags.
        client: The MLflow client. Defaults to one on the setup's backend.

    Returns:
        The run ID and the trained method run.
    """
    tags = method_tags(method=config.method, seed=seed, backbone_run_id=backbone_run_id)
    with track_mlflow(dataclasses.replace(setup, tags={**setup.tags, **tags})) as run:
        client = client if client is not None else MlflowClient()
        run_id = run.info.run_id
        method_run = train_method_run(
            backbone=load_backbone_handle(backbone_run_id, client=client),
            data=data,
            config=config,
            seed=seed,
            tracker=MLflowTracker(run_id, client=client),
            store=mlflow_checkpoint_store(run_id, client=client),
        )
    return TrackedMethodRun(run_id=run_id, run=method_run)


def evaluate_tracked_method_run(
    run_id: str,
    *,
    test_histories: pd.DataFrame,
    reference_samples: Mapping[str, np.ndarray],
    parameters: EvaluationParameters,
    client: MlflowClient | None = None,
) -> Evaluation:
    """Restore a method run from MLflow, evaluate it and log the evaluation into it.

    The method, seed and backbone (parent) come from the run's tags, the method
    configuration from its ``posterior/...`` params, and the scalers from the
    parent backbone run. The evaluation key is the run's ``evaluation_key``.

    Args:
        run_id: The method run's ID.
        test_histories: The test histories.
        reference_samples: Simulator reference samples keyed ``test_case_<i>``.
        parameters: Evaluation parameters.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        The evaluation.
    """
    client = client if client is not None else MlflowClient()
    run = client.get_run(run_id)
    method = Method(run.data.tags["method"])
    backbone_run_id = run.data.tags[PARENT_RUN_TAG]
    method_run = restore_method_run(
        backbone=load_backbone_handle(backbone_run_id, client=client),
        config=method_config(method, _unflatten(run.data.params, "posterior")),
        seed=int(run.data.tags["seed"]),
        store=mlflow_checkpoint_store(run_id, client=client),
    )
    evaluation = evaluate_predictive_model(
        method_run,
        test_histories=test_histories,
        reference_samples=reference_samples,
        x_scaler=load_scaler(backbone_run_id, "x_scaler"),
        y_scaler=load_scaler(backbone_run_id, "y_scaler"),
        parameters=parameters,
        key=method_run.evaluation_key,
    )
    log_evaluation(evaluation, MLflowTracker(run_id, client=client))
    return evaluation
