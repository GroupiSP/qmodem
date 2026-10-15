"""Tests for qmodem.hpo (seam 2: the HPO run layout on a file-backed backend)."""

from __future__ import annotations

import dataclasses
import uuid

import pytest
from mlflow.entities import Run
from stagereg.core import Split, TrainingEvent, TrainingState

from qmodem.hpo import (
    PruningEventSink,
    StudyConfig,
    StudyLayout,
    TrialRun,
    load_best_params,
    run_study,
)
from qmodem.tracking import MLFlowSetup, RunCriteria, RunSelectionError, select_run


@pytest.fixture
def setup(mlflow_backend) -> MLFlowSetup:
    """The file-backed backend with a fresh experiment for one test."""
    return dataclasses.replace(
        mlflow_backend.setup, experiment_name=f"hpo-{uuid.uuid4().hex}"
    )


def _runs(client, setup: MLFlowSetup, role: str) -> list[Run]:
    experiment = client.get_experiment_by_name(setup.experiment_name)
    return client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.role = '{role}'"
    )


def _quadratic(run: TrialRun) -> float:
    x = run.trial.suggest_float("mean/optimizer/learning_rate", -1.0, 1.0)
    run.tracker.log_metrics({"mean/validation/loss": x**2}, step=0)
    return x**2


def test_a_study_is_one_study_run_with_best_params_and_trial_children(
    mlflow_backend, setup
) -> None:
    layout = StudyLayout(stage="posterior", method="mcd", backbone_run_id="abc")

    result = run_study(
        _quadratic, config=StudyConfig(n_trials=2), layout=layout, setup=setup
    )

    client = mlflow_backend.client
    (study,) = _runs(client, setup, "hpo_study")
    assert study.info.run_id == result.run_id
    assert study.info.status == "FINISHED"
    assert {
        key: study.data.tags[key]
        for key in ("stage", "method", "backbone_run_id", "n_trials")
    } == {
        "stage": "posterior",
        "method": "mcd",
        "backbone_run_id": "abc",
        "n_trials": "2",
    }
    best_x = result.study.best_params["mean/optimizer/learning_rate"]
    assert study.data.params["best.mean/optimizer/learning_rate"] == str(best_x)
    assert result.best_value == pytest.approx(best_x**2)

    trials = _runs(client, setup, "hpo_trial")
    assert sorted(t.data.tags["hpo.trial"] for t in trials) == ["0", "1"]
    assert {t.data.tags["mlflow.parentRunId"] for t in trials} == {result.run_id}
    assert {t.info.status for t in trials} == {"FINISHED"}
    assert all("mean/validation/loss" in t.data.metrics for t in trials)


def test_tag_selection_never_returns_a_trial(mlflow_backend, setup) -> None:
    layout = StudyLayout(stage="posterior", method="mcd", backbone_run_id="abc")
    run_study(_quadratic, config=StudyConfig(n_trials=2), layout=layout, setup=setup)
    client = mlflow_backend.client
    experiment = client.get_experiment_by_name(setup.experiment_name)
    experiment_ids = [experiment.experiment_id]

    for criteria in (RunCriteria("backbone", seed=0), RunCriteria("method", 0, "mcd")):
        with pytest.raises(RunSelectionError):
            select_run(criteria, experiment_ids=experiment_ids, client=client)
    for role in ("hpo_study", "hpo_trial"):
        with pytest.raises(ValueError):
            select_run(RunCriteria(role), experiment_ids=experiment_ids, client=client)


def _report(run: TrialRun, values: list[float]) -> float:
    sink = PruningEventSink(run.trial, stage="mean")
    for epoch, value in enumerate(values):
        sink.on_event(
            TrainingEvent(
                name="validation_end",
                stage="mean",
                split=Split.VALIDATION,
                step=epoch,
                metrics={"loss": value},
                state=TrainingState(),
            )
        )
    return min(values)


def test_a_trial_worse_than_the_median_is_pruned_and_its_run_killed(
    mlflow_backend, setup
) -> None:
    histories = iter([[1.0, 0.5, 0.1], [9.0, 8.0, 7.0]])

    result = run_study(
        lambda run: _report(run, next(histories)),
        config=StudyConfig(n_trials=2, n_startup_trials=1),
        layout=StudyLayout(stage="mean"),
        setup=setup,
    )

    trials = {
        t.data.tags["hpo.trial"]: t
        for t in _runs(mlflow_backend.client, setup, "hpo_trial")
    }
    assert trials["0"].info.status == "FINISHED"
    assert trials["0"].data.tags["hpo.state"] == "complete"
    assert trials["1"].info.status == "KILLED"
    assert trials["1"].data.tags["hpo.state"] == "pruned"
    assert result.study.trials[1].intermediate_values == {0: 9.0}
    assert result.best_value == 0.1


def test_backbone_stage_studies_carry_no_method_or_backbone_tag(
    mlflow_backend, setup
) -> None:
    run_study(
        _quadratic,
        config=StudyConfig(n_trials=1),
        layout=StudyLayout("mean"),
        setup=setup,
    )

    (study,) = _runs(mlflow_backend.client, setup, "hpo_study")
    (trial,) = _runs(mlflow_backend.client, setup, "hpo_trial")
    assert {"method", "backbone_run_id"}.isdisjoint(study.data.tags)
    assert (study.data.tags["stage"], trial.data.tags["stage"]) == ("mean", "mean")
    assert "method" not in trial.data.tags


def test_trials_of_a_posterior_study_carry_its_backbone_and_seed(
    mlflow_backend, setup
) -> None:
    layout = StudyLayout("posterior", method="bbb", backbone_run_id="abc", seed=0)

    run_study(_quadratic, config=StudyConfig(n_trials=2), layout=layout, setup=setup)

    (study,) = _runs(mlflow_backend.client, setup, "hpo_study")
    trials = _runs(mlflow_backend.client, setup, "hpo_trial")
    assert study.data.tags["seed"] == "0"
    assert {(t.data.tags["backbone_run_id"], t.data.tags["seed"]) for t in trials} == {
        ("abc", "0")
    }


def test_the_best_params_of_a_study_run_are_read_back_without_prefix(
    mlflow_backend, setup
) -> None:
    layout = StudyLayout(stage="posterior", method="mcd", backbone_run_id="abc")
    result = run_study(
        _quadratic, config=StudyConfig(n_trials=3), layout=layout, setup=setup
    )

    params = load_best_params(
        result.run_id, stage="posterior", method="mcd", client=mlflow_backend.client
    )

    assert params == {
        "mean/optimizer/learning_rate": str(
            result.best_params["mean/optimizer/learning_rate"]
        )
    }


@pytest.mark.parametrize(
    ("stage", "method"), [("mean", "mcd"), ("posterior", "bbb"), ("posterior", None)]
)
def test_best_params_are_refused_from_a_study_of_another_stage_or_method(
    mlflow_backend, setup, stage, method
) -> None:
    layout = StudyLayout(stage="posterior", method="mcd", backbone_run_id="abc")
    result = run_study(
        _quadratic, config=StudyConfig(n_trials=1), layout=layout, setup=setup
    )

    with pytest.raises(ValueError):
        load_best_params(
            result.run_id, stage=stage, method=method, client=mlflow_backend.client
        )


def test_best_params_are_refused_from_a_run_that_is_not_a_study(
    mlflow_backend, make_run
) -> None:
    run_id = make_run(start_time=1, role="backbone", seed="0")

    with pytest.raises(ValueError):
        load_best_params(run_id, stage="mean", client=mlflow_backend.client)
