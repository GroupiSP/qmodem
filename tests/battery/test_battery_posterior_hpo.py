"""Tests for qmodem.battery.posterior_hpo: the posterior studies (protocol §5)."""

from __future__ import annotations

import dataclasses
import uuid

import numpy as np
import optuna
import pytest
import sklearn.preprocessing as skpp
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.data_processing import WINDOW_SIZE
from qmodem.battery.hpo import best_epoch_value
from qmodem.battery.method_run import (
    BackboneHandle,
    BBBConfig,
    CAVIConfig,
    MCDConfig,
    PSGLDConfig,
    QAVIConfig,
)
from qmodem.battery.posterior import PosteriorStageHyperparameters
from qmodem.battery.posterior_hpo import (
    POSTERIOR_SEARCH_SPACES,
    POSTERIOR_STUDY,
    Categorical,
    FloatRange,
    IntRange,
    frozen_method_config,
    posterior_objective,
    run_posterior_study,
)
from qmodem.battery.train import Method, track_backbone_run
from qmodem.hpo import StudyConfig, TrialRun
from qmodem.tracking import MLflowCheckpointStore, ScalerArtifact

optuna.logging.set_verbosity(optuna.logging.WARNING)

LR = "posterior/optimizer/learning_rate"
_AVI_SPACE = (
    FloatRange("posterior/avi/adversarial_weight", 1e-4, 1.0, log=True),
    Categorical("posterior/avi/num_draws", (1, 4, 8, 16)),
    Categorical("posterior/avi/discriminator_steps", (1, 2, 3, 5)),
    IntRange("posterior/avi/n_qubits", 3, 6),
    IntRange("posterior/avi/n_layers", 1, 4),
    FloatRange("posterior/avi/generator/learning_rate", 1e-5, 1e-2, log=True),
    FloatRange("posterior/avi/discriminator/learning_rate", 1e-5, 1e-2, log=True),
)
PROTOCOL_SPACES = {
    Method.MCD: {
        FloatRange("posterior/dropout_rate", 0.01, 0.5),
        FloatRange(LR, 1e-5, 1e-2, log=True),
    },
    Method.BBB: {
        FloatRange(LR, 1e-5, 1e-2, log=True),
        FloatRange("posterior/initial_std", 1e-4, 1e-1, log=True),
    },
    Method.PSGLD: {FloatRange("posterior/step_size", 1e-6, 1e-2, log=True)},
    Method.Q_AVI: set(_AVI_SPACE),
    Method.C_AVI: set(_AVI_SPACE),
}
"""Protocol §5's search-space table, transcribed independently."""

_SMALL_STAGE = PosteriorStageHyperparameters(max_epochs=7, patience=3)
BASES = {
    Method.MCD: MCDConfig(stage=_SMALL_STAGE),
    Method.BBB: BBBConfig(stage=_SMALL_STAGE),
    Method.PSGLD: PSGLDConfig(burn_in_epochs=2, num_samples=5, stage=_SMALL_STAGE),
    Method.Q_AVI: QAVIConfig(stage=_SMALL_STAGE),
    Method.C_AVI: CAVIConfig(stage=_SMALL_STAGE),
}
methods = st.sampled_from(sorted(PROTOCOL_SPACES))


def _random_trial(seed: int) -> optuna.Trial:
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=seed))
    return study.ask()


def test_the_posterior_study_budget_follows_the_protocol() -> None:
    assert POSTERIOR_STUDY.n_trials == 50
    assert (POSTERIOR_STUDY.n_startup_trials, POSTERIOR_STUDY.n_warmup_steps) == (5, 60)


@pytest.mark.parametrize("method", sorted(PROTOCOL_SPACES))
def test_each_method_searches_exactly_the_protocol_space(method: Method) -> None:
    space = POSTERIOR_SEARCH_SPACES[method]

    assert space.method == method
    assert set(space.dimensions) == PROTOCOL_SPACES[method]
    assert len(space.dimensions) == len(PROTOCOL_SPACES[method])


def test_hnn_has_no_study() -> None:
    assert Method.HNN not in POSTERIOR_SEARCH_SPACES


@settings(max_examples=20, deadline=None)
@given(method=methods, seed=st.integers(0, 2**31 - 1))
def test_suggestions_stay_in_the_space_and_keep_every_untuned_setting(
    method: Method, seed: int
) -> None:
    base = BASES[method]
    trial = _random_trial(seed)

    config = POSTERIOR_SEARCH_SPACES[method].suggest(trial, base)

    assert type(config) is type(base)
    assert config.stage == base.stage
    flat = _flatten(config.posterior_hyperparameters())
    for dimension in PROTOCOL_SPACES[method]:
        value = flat[dimension.name]
        if isinstance(dimension, Categorical):
            assert value in dimension.choices
        else:
            assert dimension.low <= value <= dimension.high
    untuned = {name for name in flat} - {d.name for d in PROTOCOL_SPACES[method]}
    base_flat = _flatten(base.posterior_hyperparameters())
    assert {name: flat[name] for name in untuned} == {
        name: base_flat[name] for name in untuned
    }


@settings(max_examples=20, deadline=None)
@given(method=methods, seed=st.integers(0, 2**31 - 1))
def test_logged_best_params_rebuild_the_suggested_configuration(
    method: Method, seed: int
) -> None:
    space, base = POSTERIOR_SEARCH_SPACES[method], BASES[method]
    trial = _random_trial(seed)
    suggested = space.suggest(trial, base)
    as_logged = {name: str(value) for name, value in trial.params.items()}

    assert space.apply(as_logged, base) == suggested
    assert space.apply({**as_logged, "posterior/unrelated": "1"}, base) == suggested


def test_c_avi_derives_its_hidden_width_from_the_searched_q_and_l() -> None:
    params = {"posterior/avi/n_qubits": "6", "posterior/avi/n_layers": "4"}

    q_avi = POSTERIOR_SEARCH_SPACES[Method.Q_AVI].apply(params, QAVIConfig())
    c_avi = POSTERIOR_SEARCH_SPACES[Method.C_AVI].apply(params, CAVIConfig())

    assert isinstance(c_avi, CAVIConfig)
    assert c_avi.avi == q_avi.avi
    # h = round((2·4·6 − 6) / (6 + 2)) = round(5.25) = 5 → 5·8 + 6 = 46 parameters.
    metrics = c_avi.posterior_spec().method_metrics
    assert metrics["posterior/latent_map/classical_parameters"] == 46.0


def test_a_space_refuses_another_methods_configuration() -> None:
    with pytest.raises(ValueError):
        POSTERIOR_SEARCH_SPACES[Method.MCD].apply({}, BBBConfig())


def _flatten(nested: dict, prefix: str = "posterior") -> dict:
    flat = {}
    for key, value in nested.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


# --- The objective: one trial = one method run on the seed-0 backbone (seam 1) ----

_TRIAL_STAGE = PosteriorStageHyperparameters(max_epochs=5, patience=2)
_TRIAL_BASES = {
    Method.MCD: MCDConfig(stage=_TRIAL_STAGE),
    Method.PSGLD: PSGLDConfig(burn_in_epochs=2, num_samples=4, stage=_TRIAL_STAGE),
}
_FIXED_PARAMS = {
    Method.MCD: {"posterior/dropout_rate": 0.2, LR: 1e-3},
    Method.PSGLD: {"posterior/step_size": 1e-4},
}
CRPS = "posterior/validation/crps"


def _objective(method, trained_backbone, store_factory):
    return posterior_objective(
        backbone=BackboneHandle(
            mean_hp=trained_backbone.mean_hp,
            variance_hp=trained_backbone.variance_hp,
            store=store_factory(trained_backbone.persistent),
        ),
        data=trained_backbone.battery.data,
        space=POSTERIOR_SEARCH_SPACES[method],
        base=_TRIAL_BASES[method],
    )


def _trial_run(trial, tracker_factory, store_factory) -> TrialRun:
    return TrialRun(
        trial=trial,
        run_id="trial",
        tracker=tracker_factory(),
        store=store_factory(),
    )


@pytest.fixture(scope="module")
def fixed_trials(trained_backbone, tracker_factory, store_factory):
    """Two trials per method with the same params, run through the objective."""
    results = {}
    for method in _TRIAL_BASES:
        objective = _objective(method, trained_backbone, store_factory)
        runs = [
            _trial_run(
                optuna.trial.FixedTrial(_FIXED_PARAMS[method], number=number),
                tracker_factory,
                store_factory,
            )
            for number in (0, 1)
        ]
        results[method] = [(run, objective(run)) for run in runs]
    return results


def test_an_early_stopped_trial_scores_its_best_epoch_validation_crps(
    fixed_trials,
) -> None:
    (run, value), _ = fixed_trials[Method.MCD]

    checkpoint = run.store.load("posterior/best")
    assert value == pytest.approx(min(run.tracker.history(CRPS)))
    assert value == pytest.approx(best_epoch_value(checkpoint))


def test_a_psgld_trial_scores_its_final_retained_chain(fixed_trials) -> None:
    (run, value), _ = fixed_trials[Method.PSGLD]

    history = run.tracker.history(CRPS)
    assert run.store.load("posterior/best").epoch == _TRIAL_STAGE.max_epochs - 1
    assert value == pytest.approx(history[-1])


@pytest.mark.parametrize("method", sorted(_TRIAL_BASES))
def test_trials_with_equal_params_share_the_backbone_and_key(
    fixed_trials, method, exactly_equal
) -> None:
    (first, first_value), (second, second_value) = fixed_trials[method]

    assert first_value == second_value
    exactly_equal(
        first.store.load("posterior/best").state.posterior_state,
        second.store.load("posterior/best").state.posterior_state,
    )


def test_a_trial_logs_the_suggested_configuration(fixed_trials) -> None:
    (run, _), _ = fixed_trials[Method.MCD]

    assert run.tracker.params["posterior"]["dropout_rate"] == 0.2
    assert run.tracker.params["posterior"]["optimizer"] == {"learning_rate": 1e-3}


def test_the_posterior_validation_crps_feeds_the_pruner(
    trained_backbone, tracker_factory, store_factory
) -> None:
    study = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(upper=-1.0))
    run = _trial_run(study.ask(), tracker_factory, store_factory)

    with pytest.raises(optuna.TrialPruned):
        _objective(Method.MCD, trained_backbone, store_factory)(run)

    assert len(run.tracker.history(CRPS)) == 1


# --- Posterior studies recorded in MLflow, and frozen best params (seam 2) --------

_STUDY_STAGE = PosteriorStageHyperparameters(max_epochs=3, patience=1)
_STUDY_BASES = {
    Method.MCD: MCDConfig(stage=_STUDY_STAGE),
    Method.BBB: BBBConfig(stage=_STUDY_STAGE),
    Method.PSGLD: PSGLDConfig(burn_in_epochs=1, num_samples=2, stage=_STUDY_STAGE),
    Method.Q_AVI: QAVIConfig(stage=_STUDY_STAGE),
    Method.C_AVI: CAVIConfig(stage=_STUDY_STAGE),
}


@pytest.fixture(scope="module")
def tracked_backbone_run_id(mlflow_backend, trained_backbone) -> str:
    rng = np.random.default_rng(0)
    return track_backbone_run(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        data_generation_run_id="data-generation-run",
        scalers=(
            ScalerArtifact(
                "x_scaler",
                skpp.StandardScaler().fit(rng.normal(size=(20, 2))),
                np.ones((WINDOW_SIZE, 2)),
            ),
            ScalerArtifact(
                "y_scaler",
                skpp.MinMaxScaler().fit(rng.uniform(size=(20, 1))),
                np.ones((1, 1)),
                "inverse_transform",
            ),
        ),
        setup=mlflow_backend.setup,
    ).run_id


@pytest.fixture(scope="module", params=sorted(_STUDY_BASES))
def posterior_study(request, mlflow_backend, trained_backbone, tracked_backbone_run_id):
    method = request.param
    setup = dataclasses.replace(
        mlflow_backend.setup, experiment_name=f"posterior-study-{uuid.uuid4().hex}"
    )
    result = run_posterior_study(
        backbone_run_id=tracked_backbone_run_id,
        data=trained_backbone.battery.data,
        base=_STUDY_BASES[method],
        setup=setup,
        config=StudyConfig(n_trials=2, n_startup_trials=5, n_warmup_steps=60),
        client=mlflow_backend.client,
    )
    return method, result


def test_a_posterior_study_is_a_study_run_with_trial_children_on_one_backbone(
    posterior_study, mlflow_backend, tracked_backbone_run_id
) -> None:
    method, result = posterior_study
    client = mlflow_backend.client

    study = client.get_run(result.run_id)
    trials = client.search_runs(
        [study.info.experiment_id], filter_string="tags.role = 'hpo_trial'"
    )
    expected = {
        "stage": "posterior",
        "method": method.value,
        "backbone_run_id": tracked_backbone_run_id,
        "seed": "0",
    }
    assert study.data.tags["role"] == "hpo_study"
    assert {key: study.data.tags[key] for key in expected} == expected
    assert len(trials) == 2
    for trial in trials:
        assert {key: trial.data.tags[key] for key in expected} == expected
        assert trial.data.tags["mlflow.parentRunId"] == result.run_id
        store = MLflowCheckpointStore(trial.info.run_id, client=client)
        assert store.exists("posterior/best")
        assert "mean/train/loss" not in trial.data.metrics
    tuned = {d.name for d in POSTERIOR_SEARCH_SPACES[method].dimensions}
    assert {f"best.{name}" for name in tuned} | {"best.trial"} == {
        key for key in study.data.params if key.startswith("best.")
    }


def test_a_protocol_run_is_configured_from_the_frozen_best_params(
    posterior_study, mlflow_backend
) -> None:
    method, result = posterior_study
    base = _STUDY_BASES[method]

    config = frozen_method_config(result.run_id, base, client=mlflow_backend.client)

    assert config == POSTERIOR_SEARCH_SPACES[method].apply(result.best_params, base)


def test_frozen_params_are_refused_for_another_method(
    posterior_study, mlflow_backend
) -> None:
    method, result = posterior_study
    other = Method.BBB if method != Method.BBB else Method.MCD

    with pytest.raises(ValueError):
        frozen_method_config(
            result.run_id, _STUDY_BASES[other], client=mlflow_backend.client
        )
