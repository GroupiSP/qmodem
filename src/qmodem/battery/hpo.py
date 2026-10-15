"""The battery's backbone HPO studies: the mean study and the variance study.

Protocol §5 (decided in #99 and #102): the mean study tunes the backbone sizes and
the mean learning rate, the variance study tunes the variance learning rate on
seed 0's frozen mean checkpoint. Both minimize their stage's best-epoch validation
loss (MSE, then Gamma NLL) and train through the same stage operations as backbone
runs. Trial params are named after the flattened hyperparameters
(``mean/network/kernel_size``), so a study run's ``best.*`` params map back onto
``MeanHyperparameters`` / ``VarianceHyperparameters`` with ``apply``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import optuna
from pydantic import BaseModel, ConfigDict
from stagereg.core import Checkpoint

from qmodem.hpo import PruningEventSink, StudyConfig, TrialObjective, TrialRun

from .data_processing import PreparedData
from .train import (
    MeanHyperparameters,
    VarianceHyperparameters,
    train_mean_stage,
    train_variance_stage,
)

STUDY_SEED = 0
"""The protocol seed r every backbone study trains on."""

SAMPLER_SEED = 0
"""The fixed seed of every study's TPE sampler (separate from the protocol seed)."""

MEAN_STUDY = StudyConfig(n_trials=50, sampler_seed=SAMPLER_SEED, n_startup_trials=10)
"""50 trials, TPE, median pruning with the first 10 trials never pruned."""

VARIANCE_STUDY = StudyConfig(n_trials=20, sampler_seed=SAMPLER_SEED)
"""20 trials, TPE, median pruning (Optuna's defaults: 5 startup trials)."""

Activation = Literal["gelu", "relu", "silu", "tanh"]

KERNEL_SIZE = "mean/network/kernel_size"
N_FILTERS = "mean/network/n_filters"
ACTIVATION = "mean/network/activation"
MEAN_LEARNING_RATE = "mean/optimizer/learning_rate"
VARIANCE_LEARNING_RATE = "variance/optimizer/learning_rate"


class _SearchSpace(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _apply(
    base: BaseModel, stage: str, params: Mapping[str, Any], names: tuple[str, ...]
) -> dict[str, Any]:
    """Overwrite a stage's dumped hyperparameters with the given trial params.

    Args:
        base: The stage's base hyperparameters.
        stage: The stage prefix of the param names, e.g. ``"mean"``.
        params: Trial params keyed by flattened name; values may be strings.
        names: The param names the search space owns; others are ignored.

    Returns:
        The nested hyperparameters, ready for validation.
    """
    nested = base.model_dump()
    for name in names:
        if name not in params:
            continue
        *path, leaf = name.removeprefix(f"{stage}/").split("/")
        node = nested
        for key in path:
            node = node[key]
        node[leaf] = params[name]
    return nested


class MeanSearchSpace(_SearchSpace):
    """The mean study's search space (window, batch, patience and cap fixed).

    Attributes:
        kernel_size: Inclusive kernel-size range.
        n_filters: Inclusive range of the number of filters.
        activations: Candidate activations.
        learning_rate: Mean learning-rate range, sampled on a log scale.
    """

    kernel_size: tuple[int, int] = (2, 8)
    n_filters: tuple[int, int] = (4, 64)
    activations: tuple[Activation, ...] = ("gelu", "relu", "silu", "tanh")
    learning_rate: tuple[float, float] = (1e-4, 1e-2)

    def suggest(
        self, trial: optuna.Trial, base: MeanHyperparameters
    ) -> MeanHyperparameters:
        """Sample the tuned mean hyperparameters; keep the rest of ``base``.

        Args:
            trial: The Optuna trial.
            base: The fixed settings (batch size, patience, epoch cap).

        Returns:
            The trial's mean hyperparameters.
        """
        trial.suggest_int(KERNEL_SIZE, *self.kernel_size)
        trial.suggest_int(N_FILTERS, *self.n_filters)
        trial.suggest_categorical(ACTIVATION, list(self.activations))
        trial.suggest_float(MEAN_LEARNING_RATE, *self.learning_rate, log=True)
        return self.apply(trial.params, base)

    def apply(
        self, params: Mapping[str, Any], base: MeanHyperparameters
    ) -> MeanHyperparameters:
        """Build mean hyperparameters from trial (or ``best.*``) params.

        Args:
            params: Params keyed by flattened name. Strings (MLflow params) are
                coerced; params outside this space are ignored.
            base: The fixed settings.

        Returns:
            ``base`` with the tuned values replaced.

        Raises:
            pydantic.ValidationError: If a value is invalid.
        """
        names = (KERNEL_SIZE, N_FILTERS, ACTIVATION, MEAN_LEARNING_RATE)
        return MeanHyperparameters.model_validate(_apply(base, "mean", params, names))


class VarianceSearchSpace(_SearchSpace):
    """The variance study's search space (cap and patience inherited).

    Attributes:
        learning_rate: Variance learning-rate range, sampled on a log scale.
    """

    learning_rate: tuple[float, float] = (1e-4, 1e-2)

    def suggest(
        self, trial: optuna.Trial, base: VarianceHyperparameters
    ) -> VarianceHyperparameters:
        """Sample the variance learning rate; keep the rest of ``base``.

        Args:
            trial: The Optuna trial.
            base: The fixed settings (patience, epoch cap).

        Returns:
            The trial's variance hyperparameters.
        """
        trial.suggest_float(VARIANCE_LEARNING_RATE, *self.learning_rate, log=True)
        return self.apply(trial.params, base)

    def apply(
        self, params: Mapping[str, Any], base: VarianceHyperparameters
    ) -> VarianceHyperparameters:
        """Build variance hyperparameters from trial (or ``best.*``) params.

        Args:
            params: Params keyed by flattened name. Strings (MLflow params) are
                coerced; params outside this space are ignored.
            base: The fixed settings.

        Returns:
            ``base`` with the tuned values replaced.

        Raises:
            pydantic.ValidationError: If a value is invalid.
        """
        return VarianceHyperparameters.model_validate(
            _apply(base, "variance", params, (VARIANCE_LEARNING_RATE,))
        )


def best_epoch_value(checkpoint: Checkpoint) -> float:
    """Return the best validation value a finalized stage checkpoint was chosen by.

    Args:
        checkpoint: A finalized, early-stopped stage checkpoint.

    Returns:
        The early stopper's best score (the best-epoch validation loss).

    Raises:
        ValueError: If the checkpoint carries no early-stopping score.
    """
    stopping = checkpoint.early_stopping_state
    if stopping is None or stopping.best_score is None:
        raise ValueError("The checkpoint carries no early-stopping score.")
    return float(stopping.best_score)


def mean_objective(
    *,
    data: PreparedData,
    space: MeanSearchSpace,
    base: MeanHyperparameters,
    seed: int = STUDY_SEED,
) -> TrialObjective:
    """Build the mean study's objective: best-epoch validation MSE.

    Each trial trains the mean stage alone (``train_mean_stage``), reporting the
    per-epoch validation MSE to the pruner, and keeps only ``mean/best``.

    Args:
        data: Prepared train and validation windows.
        space: The search space.
        base: The fixed mean settings.
        seed: The protocol seed the trials train on.

    Returns:
        The objective.
    """

    def objective(run: TrialRun) -> float:
        checkpoint = train_mean_stage(
            data=data,
            mean_hp=space.suggest(run.trial, base),
            seed=seed,
            tracker=run.tracker,
            store=run.store,
            event_sinks=(PruningEventSink(run.trial, stage="mean"),),
        )
        return best_epoch_value(checkpoint)

    return objective


def variance_objective(
    *,
    data: PreparedData,
    mean_hp: MeanHyperparameters,
    mean_checkpoint: Checkpoint,
    space: VarianceSearchSpace,
    base: VarianceHyperparameters,
    seed: int = STUDY_SEED,
) -> TrialObjective:
    """Build the variance study's objective: best-epoch validation Gamma NLL.

    Each trial restores the frozen mean checkpoint, trains the variance stage
    (``train_variance_stage``), reporting the per-epoch validation NLL to the
    pruner, and keeps only ``variance/best``.

    Args:
        data: Prepared train and validation windows.
        mean_hp: The mean hyperparameters of ``mean_checkpoint``.
        mean_checkpoint: The backbone's finalized ``mean/best`` checkpoint.
        space: The search space.
        base: The fixed variance settings.
        seed: The protocol seed of the backbone.

    Returns:
        The objective.
    """

    def objective(run: TrialRun) -> float:
        checkpoint = train_variance_stage(
            data=data,
            mean_hp=mean_hp,
            mean_checkpoint=mean_checkpoint,
            variance_hp=space.suggest(run.trial, base),
            seed=seed,
            tracker=run.tracker,
            store=run.store,
            event_sinks=(PruningEventSink(run.trial, stage="variance"),),
        )
        return best_epoch_value(checkpoint)

    return objective
