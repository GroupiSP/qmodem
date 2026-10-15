"""The battery's posterior HPO studies: one per method with a posterior stage.

Protocol §5 (decided in #102): every method's posterior stage gets the same budget
(50 trials, TPE with a fixed sampler seed, median pruning after 5 trials and 60
epochs) on seed 0's backbone run. Every trial restores that backbone and trains a
method run with the same key, ``fold_in(key(0), method)``, so trials differ only in
their hyperparameters. The objective is the validation CRPS of the finalized
posterior checkpoint: the best epoch's for the early-stopped methods, the final
retained chain's for pSGLD.

Search spaces are data containers (``PosteriorSearchSpace``) whose dimensions are
named after the method run's flattened posterior params
(``posterior/optimizer/learning_rate``), so a study run's ``best.*`` params map
straight back onto a method configuration with ``apply``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol

import optuna
from mlflow.tracking import MlflowClient
from stagereg.core import TrainingEvent

from qmodem.hpo import (
    PruningEventSink,
    StudyConfig,
    StudyLayout,
    StudyResult,
    TrialObjective,
    TrialRun,
    load_best_params,
    run_study,
)
from qmodem.tracking import Stage, VARIANCE_STUDY_TAG, MLFlowSetup

from .data_processing import PreparedData
from .hpo import SAMPLER_SEED, STUDY_SEED
from .method_run import (
    BackboneHandle,
    MethodConfig,
    load_backbone_handle,
    method_config,
    train_method_run,
)
from .posterior import VALIDATION_METRIC
from .train import Method

POSTERIOR_STAGE = Stage.POSTERIOR
"""The stage name of a posterior study, its trials' pruned stage and param prefix."""

POSTERIOR_STUDY = StudyConfig(
    n_trials=50, sampler_seed=SAMPLER_SEED, n_startup_trials=5, n_warmup_steps=60
)
"""50 trials, TPE, median pruning after 5 startup trials and 60 warm-up epochs."""


class Dimension(Protocol):
    """One tuned hyperparameter of a search space.

    Attributes:
        name: The flattened posterior param it sets, e.g. ``posterior/step_size``.
    """

    name: str

    def suggest(self, trial: optuna.Trial) -> Any:
        """Sample the hyperparameter in ``trial`` under ``name``.

        Args:
            trial: The Optuna trial.

        Returns:
            The sampled value.
        """
        ...


@dataclasses.dataclass(frozen=True)
class FloatRange:
    """A real hyperparameter in ``[low, high]``.

    Attributes:
        name: The flattened posterior param.
        low: Lower bound.
        high: Upper bound.
        log: Whether to sample on a log scale.
    """

    name: str
    low: float
    high: float
    log: bool = False

    def suggest(self, trial: optuna.Trial) -> float:
        """Sample the value.

        Args:
            trial: The Optuna trial.

        Returns:
            The sampled value.
        """
        return trial.suggest_float(self.name, self.low, self.high, log=self.log)


@dataclasses.dataclass(frozen=True)
class IntRange:
    """An integer hyperparameter in ``[low, high]`` (inclusive).

    Attributes:
        name: The flattened posterior param.
        low: Lower bound.
        high: Upper bound.
    """

    name: str
    low: int
    high: int

    def suggest(self, trial: optuna.Trial) -> int:
        """Sample the value.

        Args:
            trial: The Optuna trial.

        Returns:
            The sampled value.
        """
        return trial.suggest_int(self.name, self.low, self.high)


@dataclasses.dataclass(frozen=True)
class Categorical:
    """A hyperparameter chosen from a fixed set.

    Attributes:
        name: The flattened posterior param.
        choices: The candidate values.
    """

    name: str
    choices: tuple[int | float | str, ...]

    def suggest(self, trial: optuna.Trial) -> int | float | str:
        """Sample the value.

        Args:
            trial: The Optuna trial.

        Returns:
            The sampled value.
        """
        return trial.suggest_categorical(self.name, list(self.choices))


@dataclasses.dataclass(frozen=True)
class PosteriorSearchSpace:
    """One method's posterior search space.

    Attributes:
        method: The tuned method.
        dimensions: The tuned hyperparameters; every other setting comes from the
            base configuration.
    """

    method: Method
    dimensions: tuple[Dimension, ...]

    def suggest(self, trial: optuna.Trial, base: MethodConfig) -> MethodConfig:
        """Sample the tuned hyperparameters; keep the rest of ``base``.

        Args:
            trial: The Optuna trial.
            base: The fixed settings (stage schedule, pSGLD chain length, ...).

        Returns:
            The trial's method configuration.

        Raises:
            ValueError: If ``base`` configures another method.
        """
        for dimension in self.dimensions:
            dimension.suggest(trial)
        return self.apply(trial.params, base)

    def apply(self, params: Mapping[str, Any], base: MethodConfig) -> MethodConfig:
        """Build a method configuration from trial (or ``best.*``) params.

        Args:
            params: Params keyed by flattened name. Strings (MLflow params) are
                coerced; params outside this space are ignored.
            base: The fixed settings.

        Returns:
            ``base`` with the tuned values replaced.

        Raises:
            ValueError: If ``base`` configures another method.
            pydantic.ValidationError: If a value is invalid.
        """
        if base.method != self.method:
            raise ValueError(
                f"A {self.method.value} search space cannot configure "
                f"{base.method.value}."
            )
        nested = base.posterior_hyperparameters()
        for dimension in self.dimensions:
            if dimension.name not in params:
                continue
            *path, leaf = dimension.name.removeprefix(f"{POSTERIOR_STAGE}/").split("/")
            node = nested
            for key in path:
                node = node[key]
            node[leaf] = params[dimension.name]
        return method_config(self.method, nested)


_LEARNING_RATE = FloatRange("posterior/optimizer/learning_rate", 1e-5, 1e-2, log=True)

_AVI_DIMENSIONS: tuple[Dimension, ...] = (
    FloatRange("posterior/avi/adversarial_weight", 1e-4, 1.0, log=True),
    Categorical("posterior/avi/num_draws", (1, 4, 8, 16)),
    Categorical("posterior/avi/discriminator_steps", (1, 2, 3, 5)),
    IntRange("posterior/avi/n_qubits", 3, 6),
    IntRange("posterior/avi/n_layers", 1, 4),
    FloatRange("posterior/avi/generator/learning_rate", 1e-5, 1e-2, log=True),
    FloatRange("posterior/avi/discriminator/learning_rate", 1e-5, 1e-2, log=True),
)
"""Q-AVI's space, reused by C-AVI (its hidden width h derives from q and L)."""

POSTERIOR_SEARCH_SPACES: Mapping[Method, PosteriorSearchSpace] = {
    Method.MCD: PosteriorSearchSpace(
        Method.MCD,
        (FloatRange("posterior/dropout_rate", 0.01, 0.5), _LEARNING_RATE),
    ),
    Method.BBB: PosteriorSearchSpace(
        Method.BBB,
        (
            _LEARNING_RATE,
            FloatRange("posterior/initial_std", 1e-4, 1e-1, log=True),
        ),
    ),
    Method.PSGLD: PosteriorSearchSpace(
        Method.PSGLD, (FloatRange("posterior/step_size", 1e-6, 1e-2, log=True),)
    ),
    Method.Q_AVI: PosteriorSearchSpace(Method.Q_AVI, _AVI_DIMENSIONS),
    Method.C_AVI: PosteriorSearchSpace(Method.C_AVI, _AVI_DIMENSIONS),
}
"""Protocol §5's search space of every method with a posterior stage (not HNN)."""


@dataclasses.dataclass
class ValidationRecorder:
    """Stagereg ``EventSink`` keeping one validation metric of a stage per epoch.

    Attributes:
        stage: The recorded stage.
        metric: The bare validation metric name.
        values: The recorded values keyed by stage epoch.
    """

    stage: str = POSTERIOR_STAGE
    metric: str = VALIDATION_METRIC
    values: dict[int, float] = dataclasses.field(default_factory=dict)

    def on_event(self, event: TrainingEvent) -> None:
        """Record the metric of a ``validation_end`` event of ``stage``.

        Args:
            event: A training event.
        """
        if (
            event.name == "validation_end"
            and event.stage == self.stage
            and self.metric in event.metrics
        ):
            self.values[event.step] = float(event.metrics[self.metric])


def posterior_objective(
    *,
    backbone: BackboneHandle,
    data: PreparedData,
    space: PosteriorSearchSpace,
    base: MethodConfig,
    seed: int = STUDY_SEED,
) -> TrialObjective:
    """Build a posterior study's objective: the finalized epoch's validation CRPS.

    Each trial trains one method run (``train_method_run``) on the restored
    backbone with ``method_key(seed, method)``, reporting the per-epoch validation
    CRPS to the pruner, and keeps only ``posterior/best``. The finalized epoch is
    the best one for the early-stopped methods and the last one for pSGLD, whose
    value is then the CRPS of the final retained chain.

    Args:
        backbone: The seed-0 backbone every trial restores.
        data: Prepared train and validation windows.
        space: The method's search space.
        base: The method's fixed settings.
        seed: The protocol seed of the backbone (and of the trials' key).

    Returns:
        The objective.
    """

    def objective(run: TrialRun) -> float:
        """Train one method run with the trial's posterior HPs.

        Args:
            run: The trial and its run's tracker and store.

        Returns:
            The validation CRPS at the finalized epoch.

        Raises:
            ValueError: If the method has no posterior stage.
        """
        recorder = ValidationRecorder()
        method_run = train_method_run(
            backbone=backbone,
            data=data,
            config=space.suggest(run.trial, base),
            seed=seed,
            tracker=run.tracker,
            store=run.store,
            event_sinks=(
                recorder,
                PruningEventSink(run.trial, POSTERIOR_STAGE, VALIDATION_METRIC),
            ),
        )
        if method_run.checkpoint is None:
            raise ValueError(f"{space.method.value} has no posterior stage to tune.")
        return recorder.values[method_run.checkpoint.epoch]

    return objective


def search_space(method: Method) -> PosteriorSearchSpace:
    """Return a method's posterior search space.

    Args:
        method: The method.

    Returns:
        Its search space.

    Raises:
        ValueError: If the method has no posterior study (HNN).
    """
    if method not in POSTERIOR_SEARCH_SPACES:
        raise ValueError(f"{method.value} has no posterior study.")
    return POSTERIOR_SEARCH_SPACES[method]


def require_tuned_backbone(
    backbone_run_id: str, *, client: MlflowClient | None = None
) -> None:
    """Refuse a backbone whose variance LR was not set by the variance study.

    Posterior studies run on the seed-0 backbone re-trained from both the mean and
    the variance study (protocol §5), so their trials see the tuned aleatoric
    variance.

    Args:
        backbone_run_id: The backbone run hosting the study.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Raises:
        ValueError: If the backbone has no ``VARIANCE_STUDY_TAG``.
    """
    client = client if client is not None else MlflowClient()
    if VARIANCE_STUDY_TAG not in client.get_run(backbone_run_id).data.tags:
        raise ValueError(
            f"Backbone run {backbone_run_id} was not trained from a variance study "
            f"(no {VARIANCE_STUDY_TAG!r} tag); re-train it with "
            "--mean-study-run-id and --variance-study-run-id first."
        )


def run_posterior_study(
    *,
    backbone_run_id: str,
    data: PreparedData,
    base: MethodConfig,
    setup: MLFlowSetup,
    config: StudyConfig = POSTERIOR_STUDY,
    seed: int = STUDY_SEED,
    storage: optuna.storages.BaseStorage | str | None = None,
    client: MlflowClient | None = None,
) -> StudyResult:
    """Run one method's posterior study on a backbone run recorded in MLflow.

    The study run is tagged ``stage=posterior``, ``method``, ``backbone_run_id``,
    ``seed`` and ``n_trials``; each trial is a ``role=hpo_trial`` child with the
    same tags, holding its method run's params, posterior metrics and
    ``checkpoints/posterior/best``.

    Args:
        backbone_run_id: The seed-0 backbone run every trial restores.
        data: Prepared train and validation windows.
        base: The method's fixed settings (its type selects the method).
        setup: Tracking backend and experiment.
        config: The study budget.
        seed: The protocol seed of the backbone (and of the trials' key).
        storage: Optuna storage. ``None`` keeps the study in memory.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        The study run's ID and the Optuna study.

    Raises:
        ValueError: If the method has no posterior study (HNN).
    """
    space = search_space(base.method)
    return run_study(
        posterior_objective(
            backbone=load_backbone_handle(backbone_run_id, client=client),
            data=data,
            space=space,
            base=base,
            seed=seed,
        ),
        config=config,
        layout=StudyLayout(
            stage=POSTERIOR_STAGE,
            method=base.method.value,
            backbone_run_id=backbone_run_id,
            seed=seed,
        ),
        setup=setup,
        storage=storage,
    )


def frozen_method_config(
    study_run_id: str, base: MethodConfig, client: MlflowClient | None = None
) -> MethodConfig:
    """Configure a protocol method run from a posterior study's frozen ``best.*``.

    Args:
        study_run_id: The method's ``role=hpo_study``, ``stage=posterior`` run.
        base: The fixed settings the study ran with (stage schedule etc.).
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        ``base`` with the study's selected hyperparameters.

    Raises:
        ValueError: If the run is not a posterior study of ``base``'s method.
    """
    params = load_best_params(
        study_run_id, stage=POSTERIOR_STAGE, method=base.method.value, client=client
    )
    return search_space(base.method).apply(params, base)
