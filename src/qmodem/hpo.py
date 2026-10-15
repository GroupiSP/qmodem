"""Optuna studies laid out as MLflow runs (protocol §5).

A study is one top-level ``role=hpo_study`` run. Each Optuna trial is a
``role=hpo_trial`` child run, tagged ``hpo.trial=<n>``, into which the objective logs
its stage's metrics and checkpoints. When the study ends, the study run logs the
selected hyperparameters as ``best.<name>`` params. The module is dataset-agnostic:
objectives (mean, variance and posterior stages) are supplied by the caller.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

import optuna
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from stagereg.core import CheckpointStore, ExperimentTracker, TrainingEvent

from .tracking import (
    MLFlowSetup,
    MLflowTracker,
    hpo_study_tags,
    hpo_trial_tags,
    mlflow_checkpoint_store,
    track_mlflow,
)

HPO_STATE_TAG = "hpo.state"
BEST_PARAM_PREFIX = "best."
BEST_VALUE_METRIC = "best.value"


@dataclasses.dataclass(frozen=True)
class StudyConfig:
    """The budget of one study: TPE with a fixed seed and median pruning.

    Attributes:
        n_trials: Number of trials.
        sampler_seed: Seed of the ``TPESampler``.
        n_startup_trials: Trials the ``MedianPruner`` never prunes.
        n_warmup_steps: Steps (epochs) of every trial the pruner never prunes.
    """

    n_trials: int
    sampler_seed: int = 0
    n_startup_trials: int = 5
    n_warmup_steps: int = 0


@dataclasses.dataclass(frozen=True)
class StudyLayout:
    """What a study tunes, recorded as the study and trial runs' tags.

    Attributes:
        stage: The tuned stage (``"mean"``, ``"variance"`` or ``"posterior"``).
        method: The tuned method, or ``None`` for a backbone stage.
        backbone_run_id: The backbone run every trial restores, or ``None``.
    """

    stage: str
    method: str | None = None
    backbone_run_id: str | None = None


@dataclasses.dataclass(frozen=True)
class TrialRun:
    """One trial and the MLflow child run it records into.

    Attributes:
        trial: The Optuna trial (suggests hyperparameters, receives reports).
        run_id: The trial's ``role=hpo_trial`` run.
        tracker: Logs the trial's params and stage metrics into its run.
        store: Keeps the trial's checkpoints; only finalized ones reach its run.
    """

    trial: optuna.Trial
    run_id: str
    tracker: ExperimentTracker
    store: CheckpointStore


TrialObjective = Callable[[TrialRun], float]
"""Trains one trial's stage and returns the value the study minimizes."""


@dataclasses.dataclass(frozen=True)
class StudyResult:
    """A finished study.

    Attributes:
        run_id: The ``role=hpo_study`` run.
        study: The Optuna study.
    """

    run_id: str
    study: optuna.Study

    @property
    def best_params(self) -> dict[str, Any]:
        """The best trial's hyperparameters, keyed by their suggestion name."""
        return self.study.best_params

    @property
    def best_value(self) -> float:
        """The best trial's objective value."""
        return self.study.best_value


def make_study(
    config: StudyConfig, *, storage: optuna.storages.BaseStorage | str | None = None
) -> optuna.Study:
    """Create a minimizing study with a seeded TPE sampler and a median pruner.

    Args:
        config: The study budget.
        storage: Optuna storage. ``None`` keeps the study in memory.

    Returns:
        The new study.
    """
    return optuna.create_study(
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.sampler_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.n_startup_trials,
            n_warmup_steps=config.n_warmup_steps,
        ),
    )


@dataclasses.dataclass(frozen=True)
class PruningEventSink:
    """Stagereg ``EventSink`` that reports a validation metric to an Optuna trial.

    On every ``validation_end`` event of ``stage`` it reports ``metric`` at the
    stage epoch, and raises ``optuna.TrialPruned`` if the pruner says so, which
    aborts the stage before it finalizes a checkpoint.

    Attributes:
        trial: The trial receiving the reports.
        stage: The stage whose validation metric is reported, e.g. ``"mean"``.
        metric: The bare validation metric name, e.g. ``"loss"``.
    """

    trial: optuna.Trial
    stage: str
    metric: str = "loss"

    def on_event(self, event: TrainingEvent) -> None:
        """Report the stage's validation metric and prune if told to.

        Args:
            event: A training event.

        Raises:
            optuna.TrialPruned: If the pruner prunes the trial at this epoch.
        """
        if (
            event.name != "validation_end"
            or event.stage != self.stage
            or self.metric not in event.metrics
        ):
            return
        self.trial.report(float(event.metrics[self.metric]), step=event.step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at {self.stage} epoch {event.step}.")


def run_study(
    objective: TrialObjective,
    *,
    config: StudyConfig,
    layout: StudyLayout,
    setup: MLFlowSetup,
    storage: optuna.storages.BaseStorage | str | None = None,
) -> StudyResult:
    """Run a study as one ``role=hpo_study`` run with ``role=hpo_trial`` children.

    Each trial gets a child run tagged ``hpo.trial=<n>``, an ``MLflowTracker`` and a
    production checkpoint store on it, so only the checkpoints its stage finalizes
    are uploaded. A completed trial run ends ``FINISHED``, a pruned one ``KILLED``
    and a failed one ``FAILED``; each carries its Optuna state as ``hpo.state``.
    The study run logs the best trial's params as ``best.<name>``, its number as
    ``best.trial`` and its value as the ``best.value`` metric.

    Args:
        objective: Trains one trial and returns the value to minimize.
        config: The study budget.
        layout: The tuned stage, method and backbone run.
        setup: Tracking backend and experiment. Its tags are extended with the
            study tags.
        storage: Optuna storage. ``None`` keeps the study in memory.

    Returns:
        The study run's ID and the Optuna study.

    Raises:
        ValueError: If no trial completed.
    """
    study = make_study(config, storage=storage)
    tags = hpo_study_tags(
        stage=layout.stage,
        n_trials=config.n_trials,
        method=layout.method,
        backbone_run_id=layout.backbone_run_id,
    )
    with track_mlflow(dataclasses.replace(setup, tags={**setup.tags, **tags})) as run:
        study_run_id = run.info.run_id
        client = MlflowClient()

        def run_trial(trial: optuna.Trial) -> float:
            return _run_trial(
                trial,
                objective,
                client=client,
                experiment_id=run.info.experiment_id,
                study_run_id=study_run_id,
                layout=layout,
            )

        study.optimize(run_trial, n_trials=config.n_trials)
        _log_best(client, study_run_id, study.best_trial)
    return StudyResult(run_id=study_run_id, study=study)


def _run_trial(
    trial: optuna.Trial,
    objective: TrialObjective,
    *,
    client: MlflowClient,
    experiment_id: str,
    study_run_id: str,
    layout: StudyLayout,
) -> float:
    """Run one trial inside its own child run.

    Args:
        trial: The Optuna trial.
        objective: The study objective.
        client: The MLflow client.
        experiment_id: The study's experiment.
        study_run_id: The parent study run.
        layout: The tuned stage and method.

    Returns:
        The trial's objective value.
    """
    run_id = client.create_run(
        experiment_id,
        run_name=f"trial-{trial.number}",
        tags=hpo_trial_tags(
            study_run_id=study_run_id,
            number=trial.number,
            stage=layout.stage,
            method=layout.method,
        ),
    ).info.run_id
    trial_run = TrialRun(
        trial=trial,
        run_id=run_id,
        tracker=MLflowTracker(run_id, client=client),
        store=mlflow_checkpoint_store(run_id, client=client),
    )
    try:
        value = objective(trial_run)
    except optuna.TrialPruned:
        _end_trial_run(client, run_id, "KILLED", optuna.trial.TrialState.PRUNED)
        raise
    except BaseException:
        _end_trial_run(client, run_id, "FAILED", optuna.trial.TrialState.FAIL)
        raise
    _end_trial_run(client, run_id, "FINISHED", optuna.trial.TrialState.COMPLETE)
    return value


def _end_trial_run(
    client: MlflowClient, run_id: str, status: str, state: optuna.trial.TrialState
) -> None:
    """Record a trial's Optuna state and terminate its run.

    Args:
        client: The MLflow client.
        run_id: The trial run.
        status: The MLflow terminal status.
        state: The trial's Optuna state.
    """
    client.set_tag(run_id, HPO_STATE_TAG, state.name.lower())
    client.set_terminated(run_id, status=status)


def _log_best(
    client: MlflowClient, run_id: str, best: optuna.trial.FrozenTrial
) -> None:
    """Log the best trial's params, number and value into the study run.

    Args:
        client: The MLflow client.
        run_id: The study run.
        best: The best trial.
    """
    params: Mapping[str, Any] = {
        **{f"{BEST_PARAM_PREFIX}{key}": value for key, value in best.params.items()},
        f"{BEST_PARAM_PREFIX}trial": best.number,
    }
    client.log_batch(
        run_id,
        params=[Param(key, str(value)) for key, value in params.items()],
        metrics=[
            Metric(BEST_VALUE_METRIC, float(best.value), int(time.time() * 1000), 0)
        ],
    )
