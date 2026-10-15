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
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from mlflow.tracking import MlflowClient
from stagereg.core import (
    Batch,
    Checkpoint,
    CheckpointStore,
    EventSink,
    ExperimentTracker,
    TrackerEventSink,
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

from .data_processing import PreparedData, make_loader, posterior_dataset_size
from .evaluate import (
    Evaluation,
    EvaluationParameters,
    evaluate_predictive_model,
    log_evaluation,
)
from .methods.avi import CAVIConfig, QAVIConfig
from .methods.base import MethodConfig
from .methods.bbb import BBBConfig
from .methods.hnn import HNNConfig
from .methods.mcd import MCDConfig
from .methods.psgld import PSGLDConfig
from .models import MeanNet, VarianceNet
from .posterior import (
    POSTERIOR_CHECKPOINT_KEY,
    PREDICTIVE_SAMPLE_COUNT,
    EpochTimer,
    PosteriorCost,
    PosteriorKeys,
    build_posterior_stage,
    draws_per_second,
    log_cost,
    parameter_count,
)
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


METHOD_CONFIGS: Mapping[Method, type[MethodConfig]] = {
    Method.HNN: HNNConfig,
    Method.MCD: MCDConfig,
    Method.BBB: BBBConfig,
    Method.PSGLD: PSGLDConfig,
    Method.Q_AVI: QAVIConfig,
    Method.C_AVI: CAVIConfig,
}
"""The configuration type of every implemented method (strategy registry).

A new method adds its configuration module under ``qmodem.battery.methods`` and
one entry here; the pipeline needs no change.
"""


DATASET_SIZE_PARAM = "dataset_size"
"""Name of the logged posterior param holding the dataset size N.

It is a method-run parameter, not part of any method configuration, so
``method_config`` ignores it.
"""


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
    configuration = {
        key: value
        for key, value in hyperparameters.items()
        if key != DATASET_SIZE_PARAM
    }
    return METHOD_CONFIGS[method].model_validate(configuration)


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

        Without a posterior (HNN), every draw is the backbone's single mean. A
        finite posterior (pSGLD) always uses all its retained draws, so
        ``num_draws`` does not apply to it.

        Args:
            inputs: Windows ``(B, W, C)``.
            key: Key of the posterior draws.
            num_draws: Number of draws M of an unlimited posterior.

        Returns:
            The draws' means ``(M, B, 1)`` with the aleatoric variance ``(B, 1)``.
        """
        if self.posterior is None:
            mean = self.mean_net(inputs)
            draws = jnp.broadcast_to(mean, (num_draws, *mean.shape))
        elif self.posterior.num_draws is not None:
            draws = self.posterior.sample_means(inputs, key)
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


def _validation_inputs(data: PreparedData) -> jax.Array:
    """Return every validation window, the inputs of the draws-per-second probe.

    Args:
        data: Prepared train and validation windows.

    Returns:
        The validation windows ``(B, W, C)``.
    """
    return jnp.asarray(data.val.features)


def _cost(
    run: MethodRun,
    *,
    data: PreparedData,
    wall_clock_seconds: float,
    trainable_parameters: int,
) -> PosteriorCost:
    """Measure the cost of a trained method run (table T3).

    Args:
        run: The trained method run.
        data: Its windows (the draws are timed at the validation windows).
        wall_clock_seconds: The posterior stage's wall-clock time.
        trainable_parameters: The method's trainable parameter count.

    Returns:
        The cost, with the draws per second of ``PREDICTIVE_SAMPLE_COUNT`` draws.
    """
    inputs = _validation_inputs(data)
    key = run.evaluation_key
    finite_draws = None if run.posterior is None else run.posterior.num_draws
    return PosteriorCost(
        wall_clock_seconds=wall_clock_seconds,
        trainable_parameters=trainable_parameters,
        draws_per_second=draws_per_second(
            lambda: run.predictive(inputs, key, PREDICTIVE_SAMPLE_COUNT).draws,
            finite_draws or PREDICTIVE_SAMPLE_COUNT,
        ),
    )


def train_method_run(
    *,
    backbone: BackboneHandle,
    data: PreparedData,
    config: MethodConfig,
    seed: int,
    tracker: ExperimentTracker,
    store: CheckpointStore,
    dataset_size: int | None = None,
    event_sinks: Sequence[EventSink] = (),
) -> MethodRun:
    """Train a method run on a restored backbone.

    The backbone's finalized ``mean`` and ``variance`` checkpoints are restored,
    never re-trained. The training state is then re-keyed with
    ``method_key(seed, method)``. A method with a posterior stage then trains the
    shared posterior stage (``build_posterior_stage``) and finalizes its best
    epoch on validation CRPS into ``store`` (``posterior/best``). HNN trains and
    saves nothing. Every method run logs its posterior hyperparameters (with the
    dataset size N), its method metrics (``PosteriorSpec.method_metrics``) and its
    cost metrics (``cost/posterior/...``).

    Args:
        backbone: The backbone run to restore.
        data: Prepared train and validation windows (for the posterior stage).
        config: The method's configuration.
        seed: The protocol seed r.
        tracker: Receives the posterior hyperparameters, stage metrics and cost.
        store: Receives the method run's ``posterior`` checkpoints.
        dataset_size: The posterior's dataset size N. Defaults to the number of
            training windows.
        event_sinks: Extra receivers of the posterior stage's events, e.g. an
            HPO pruner. Unused by HNN.

    Returns:
        The method run, with its finalized ``posterior`` checkpoint (``None`` for
        HNN).

    Raises:
        NotImplementedError: If the method is not implemented yet.
    """
    _require_registered(config)
    spec = config.posterior_spec()
    hyperparameters = config.posterior_hyperparameters()
    if spec is not None:
        if dataset_size is None:
            dataset_size = posterior_dataset_size(data)
        hyperparameters = {**hyperparameters, DATASET_SIZE_PARAM: dataset_size}
    if hyperparameters:
        tracker.log_params({"posterior": hyperparameters})
    if spec is not None and spec.method_metrics:
        tracker.log_metrics(spec.method_metrics, step=0)
    state, mean_net, variance_net = _restore_backbone(backbone, seed)
    key = method_key(seed, config.method)
    state.rng_state = key
    if spec is None:
        run = MethodRun(
            method=config.method, key=key, mean_net=mean_net, variance_net=variance_net
        )
        log_cost(
            tracker,
            _cost(run, data=data, wall_clock_seconds=0.0, trainable_parameters=0),
        )
        return run

    keys = PosteriorKeys.of(key)
    method = spec.build_method(keys.init)
    timer = EpochTimer(tracker)
    stage = build_posterior_stage(
        spec=spec,
        method=method,
        network=backbone.mean_hp.network,
        keys=keys,
        loader=make_loader(data, batch_size=spec.stage.batch_size, key=keys.loader),
        dataset_size=dataset_size,
        event_sinks=(TrackerEventSink(tracker), timer, *event_sinks),
        store=store,
    )
    start = time.perf_counter()
    stage.prepare(state)
    timer.start()
    stage.train(state)
    wall_clock_seconds = time.perf_counter() - start
    timer.log_mean()
    trainable_parameters = parameter_count(spec.trainable_parameters(method))
    checkpoint = store.load(POSTERIOR_CHECKPOINT_KEY)
    # Rebuild the posterior from the finalized checkpoint, exactly as a restore
    # does: BBB's stds -> rhos -> stds round trip is not bit-exact, and the run
    # must predict like its restored self.
    method.load_posterior(checkpoint.state.posterior_state)
    run = MethodRun(
        method=config.method,
        key=key,
        mean_net=mean_net,
        variance_net=variance_net,
        posterior=method.posterior(),
        checkpoint=checkpoint,
    )
    log_cost(
        tracker,
        _cost(
            run,
            data=data,
            wall_clock_seconds=wall_clock_seconds,
            trainable_parameters=trainable_parameters,
        ),
    )
    return run


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
    state, mean_net, variance_net = _restore_backbone(backbone, seed)
    key = method_key(seed, config.method)
    spec = config.posterior_spec()
    if spec is None:
        return MethodRun(
            method=config.method, key=key, mean_net=mean_net, variance_net=variance_net
        )
    keys = PosteriorKeys.of(key)
    method = spec.build_method(keys.init)
    checkpoint = store.load(POSTERIOR_CHECKPOINT_KEY)
    build_posterior_stage(
        spec=spec, method=method, network=backbone.mean_hp.network, keys=keys
    ).restore(state, checkpoint)
    return MethodRun(
        method=config.method,
        key=key,
        mean_net=mean_net,
        variance_net=variance_net,
        posterior=method.posterior(),
        checkpoint=checkpoint,
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
    dataset_size: int | None = None,
) -> TrackedMethodRun:
    """Train a method run inside an MLflow run, as a child of its backbone run.

    The run is tagged with ``method_tags``. It receives the posterior
    hyperparameters (flattened under ``posterior``, with the dataset size N), the
    posterior-stage metrics, the cost metrics and the finalized
    ``checkpoints/posterior/best`` (none for HNN). The backbone is restored from
    the parent run's checkpoints.

    Args:
        backbone_run_id: The backbone run's ID.
        data: Prepared train and validation windows.
        config: The method's configuration.
        seed: The protocol seed r (should match the backbone's ``seed`` tag).
        setup: Tracking backend, experiment and run name. Its tags are extended
            with the method-run tags.
        client: The MLflow client. Defaults to one on the setup's backend.
        dataset_size: The posterior's dataset size N. Defaults to the number of
            training windows.

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
            dataset_size=dataset_size,
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
