"""Tests for qmodem.battery.hpo: the mean and variance studies (protocol §5)."""

from __future__ import annotations

import dataclasses
import math
import uuid

import optuna
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.data_processing import ValidationSet
from qmodem.battery.hpo import (
    MEAN_STUDY,
    VARIANCE_STUDY,
    MeanSearchSpace,
    VarianceSearchSpace,
    frozen_backbone_hyperparameters,
    mean_objective,
    variance_objective,
)
from qmodem.battery.train import MeanHyperparameters, VarianceHyperparameters
from qmodem.hpo import StudyConfig, StudyLayout, TrialRun, run_study
from qmodem.tracking import MLflowCheckpointStore

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _random_trial(seed: int) -> optuna.Trial:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    return study.ask()


def test_the_study_budgets_follow_the_protocol() -> None:
    assert (MEAN_STUDY.n_trials, MEAN_STUDY.n_startup_trials) == (50, 10)
    assert VARIANCE_STUDY.n_trials == 20
    assert MEAN_STUDY.sampler_seed == VARIANCE_STUDY.sampler_seed


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_mean_suggestions_stay_in_the_protocol_space_and_keep_the_fixed_settings(
    seed: int,
) -> None:
    base = MeanHyperparameters()

    hp = MeanSearchSpace().suggest(_random_trial(seed), base)

    assert 2 <= hp.network.kernel_size <= 8
    assert 4 <= hp.network.n_filters <= 64
    assert hp.network.activation in {"gelu", "relu", "silu", "tanh"}
    assert 1e-4 <= hp.optimizer.learning_rate <= 1e-2
    assert (hp.batch_size, hp.schedule) == (32, base.schedule)
    assert (hp.schedule.patience, hp.schedule.max_epochs) == (50, 500)


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_variance_suggestions_stay_in_the_protocol_space(seed: int) -> None:
    base = VarianceHyperparameters()

    hp = VarianceSearchSpace().suggest(_random_trial(seed), base)

    assert 1e-4 <= hp.optimizer.learning_rate <= 1e-2
    assert hp.schedule == base.schedule


@settings(max_examples=30, deadline=None)
@given(seed=st.integers(0, 2**31 - 1))
def test_trial_params_rebuild_the_suggested_hyperparameters(seed: int) -> None:
    trial = _random_trial(seed)
    mean_space, variance_space = MeanSearchSpace(), VarianceSearchSpace()
    mean_base, variance_base = MeanHyperparameters(), VarianceHyperparameters()

    mean_hp = mean_space.suggest(trial, mean_base)
    variance_hp = variance_space.suggest(trial, variance_base)

    assert mean_space.apply(trial.params, mean_base) == mean_hp
    assert variance_space.apply(trial.params, variance_base) == variance_hp


# --- Studies on the synthetic battery, recorded in MLflow (seam 2) -----------------

_TINY_MEAN_SPACE = MeanSearchSpace(kernel_size=(2, 3), n_filters=(2, 4))
_TWO_TRIALS = StudyConfig(n_trials=2, n_startup_trials=10)


@pytest.fixture
def setup(mlflow_backend):
    """The file-backed backend with a fresh experiment for one test."""
    return dataclasses.replace(
        mlflow_backend.setup, experiment_name=f"study-{uuid.uuid4().hex}"
    )


def _trial_runs(client, result):
    experiment_id = client.get_run(result.run_id).info.experiment_id
    return client.search_runs([experiment_id], filter_string="tags.role = 'hpo_trial'")


def test_mean_and_variance_trials_score_the_backbone_validation_set(
    trained_backbone, only_serving, tracker_factory, store_factory
) -> None:
    data = only_serving(trained_backbone.battery.data, ValidationSet.BACKBONE)
    objectives = (
        mean_objective(
            data=data, space=_TINY_MEAN_SPACE, base=trained_backbone.mean_hp
        ),
        variance_objective(
            data=data,
            mean_hp=trained_backbone.mean_hp,
            mean_checkpoint=trained_backbone.checkpoints.mean,
            space=VarianceSearchSpace(),
            base=trained_backbone.variance_hp,
        ),
    )

    for objective in objectives:
        run = TrialRun(
            trial=_random_trial(0),
            run_id="trial",
            tracker=tracker_factory(),
            store=store_factory(),
        )
        assert math.isfinite(objective(run))


def _min_history(client, run_id: str, tag: str) -> float:
    return min(m.value for m in client.get_metric_history(run_id, tag))


def test_a_mean_study_trains_and_keeps_only_mean_checkpoints(
    mlflow_backend, setup, trained_backbone
) -> None:
    result = run_study(
        mean_objective(
            data=trained_backbone.battery.data,
            space=_TINY_MEAN_SPACE,
            base=trained_backbone.mean_hp,
        ),
        config=_TWO_TRIALS,
        layout=StudyLayout(stage="mean"),
        setup=setup,
    )

    client = mlflow_backend.client
    trials = _trial_runs(client, result)
    assert len(trials) == 2
    for trial in trials:
        store = MLflowCheckpointStore(trial.info.run_id, client=client)
        assert store.exists("mean/best")
        assert not store.exists("variance/best")
    best = min(
        _min_history(client, t.info.run_id, "mean/validation/loss") for t in trials
    )
    assert result.best_value == pytest.approx(best)
    params = client.get_run(result.run_id).data.params
    rebuilt = _TINY_MEAN_SPACE.apply(
        {k.removeprefix("best."): v for k, v in params.items() if k != "best.trial"},
        trained_backbone.mean_hp,
    )
    assert rebuilt == _TINY_MEAN_SPACE.apply(
        result.best_params, trained_backbone.mean_hp
    )


def test_a_variance_study_trains_on_the_frozen_mean_and_keeps_only_variance(
    mlflow_backend, setup, trained_backbone
) -> None:
    result = run_study(
        variance_objective(
            data=trained_backbone.battery.data,
            mean_hp=trained_backbone.mean_hp,
            mean_checkpoint=trained_backbone.checkpoints.mean,
            space=VarianceSearchSpace(),
            base=trained_backbone.variance_hp,
            seed=trained_backbone.seed,
        ),
        config=_TWO_TRIALS,
        layout=StudyLayout(stage="variance", backbone_run_id="backbone"),
        setup=setup,
    )

    client = mlflow_backend.client
    trials = _trial_runs(client, result)
    for trial in trials:
        store = MLflowCheckpointStore(trial.info.run_id, client=client)
        assert store.exists("variance/best")
        assert not store.exists("mean/best")
        assert "mean/train/loss" not in trial.data.metrics
    best = min(
        _min_history(client, t.info.run_id, "variance/validation/loss") for t in trials
    )
    assert result.best_value == pytest.approx(best)


@pytest.fixture(scope="module")
def backbone_studies(mlflow_backend, trained_backbone):
    """A two-trial mean study and a two-trial variance study, in one experiment."""
    setup = dataclasses.replace(
        mlflow_backend.setup, experiment_name=f"backbone-{uuid.uuid4().hex}"
    )
    data = trained_backbone.battery.data
    mean = run_study(
        mean_objective(
            data=data, space=MeanSearchSpace(), base=trained_backbone.mean_hp
        ),
        config=_TWO_TRIALS,
        layout=StudyLayout(stage="mean"),
        setup=setup,
    )
    variance = run_study(
        variance_objective(
            data=data,
            mean_hp=trained_backbone.mean_hp,
            mean_checkpoint=trained_backbone.checkpoints.mean,
            space=VarianceSearchSpace(),
            base=trained_backbone.variance_hp,
        ),
        config=_TWO_TRIALS,
        layout=StudyLayout(stage="variance", backbone_run_id="backbone"),
        setup=setup,
    )
    return mean, variance


def test_a_protocol_backbone_is_configured_from_both_studies_frozen_params(
    backbone_studies, mlflow_backend, trained_backbone
) -> None:
    mean, variance = backbone_studies

    mean_hp, variance_hp = frozen_backbone_hyperparameters(
        mean_study_run_id=mean.run_id,
        variance_study_run_id=variance.run_id,
        mean_base=trained_backbone.mean_hp,
        variance_base=trained_backbone.variance_hp,
        client=mlflow_backend.client,
    )

    assert mean_hp == MeanSearchSpace().apply(
        mean.best_params, trained_backbone.mean_hp
    )
    assert variance_hp == VarianceSearchSpace().apply(
        variance.best_params, trained_backbone.variance_hp
    )
    assert mean_hp.schedule == trained_backbone.mean_hp.schedule
    assert variance_hp.schedule == trained_backbone.variance_hp.schedule


def test_swapped_backbone_studies_are_refused(
    backbone_studies, mlflow_backend, trained_backbone
) -> None:
    mean, variance = backbone_studies

    with pytest.raises(ValueError):
        frozen_backbone_hyperparameters(
            mean_study_run_id=variance.run_id,
            variance_study_run_id=mean.run_id,
            mean_base=trained_backbone.mean_hp,
            variance_base=trained_backbone.variance_hp,
            client=mlflow_backend.client,
        )


def test_the_variance_study_backbone_takes_the_frozen_mean_and_default_variance(
    backbone_studies, mlflow_backend, trained_backbone
) -> None:
    mean, _ = backbone_studies

    mean_hp, variance_hp = frozen_backbone_hyperparameters(
        mean_study_run_id=mean.run_id,
        variance_study_run_id=None,
        mean_base=trained_backbone.mean_hp,
        variance_base=trained_backbone.variance_hp,
        client=mlflow_backend.client,
    )

    assert mean_hp == MeanSearchSpace().apply(
        mean.best_params, trained_backbone.mean_hp
    )
    assert variance_hp == trained_backbone.variance_hp
