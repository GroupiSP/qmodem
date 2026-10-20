"""Tests for qmodem.tracking (seam 2: MLflow adapters on a file-backed backend)."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import pathlib

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pytest
from flax import nnx
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from matplotlib.figure import Figure
from stagereg.core import (
    Checkpoint,
    EarlyStoppingState,
    StageState,
    TrainingState,
    flatten_parameters,
    is_finalized,
)

from qmodem.battery.train import MeanNetworkHyperparameters, build_mean_net
from qmodem.tracking import (
    PROTOCOL_OVERRIDES_PARAMS_TAG,
    PROTOCOL_OVERRIDES_TAG,
    MLflowCheckpointStore,
    MLflowTracker,
    RunCriteria,
    RunSelectionError,
    RunSelector,
    backbone_tags,
    mlflow_checkpoint_store,
    protocol_override_tags,
    resolve_run_id,
    stagereg_version,
    track_mlflow,
    unflatten_parameters,
)

_MLFLOW_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12)


def test_stagereg_version_is_the_installed_distribution_version() -> None:
    assert stagereg_version() == importlib.metadata.version("stagereg") == "0.3.0"


@_MLFLOW_SETTINGS
@given(
    stage=_names,
    split=st.sampled_from(["train", "validation"]),
    metric=_names,
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, width=32),
        min_size=1,
        max_size=5,
    ),
)
def test_metric_tags_arrive_verbatim_with_the_stage_epoch_as_step(
    mlflow_backend, stage: str, split: str, metric: str, values: list[float]
) -> None:
    run_id = mlflow_backend.new_run()
    tracker = MLflowTracker(run_id, client=mlflow_backend.client)
    tag = f"{stage}/{split}/{metric}"

    for epoch, value in enumerate(values):
        tracker.log_metrics({tag: value}, step=epoch)

    history = mlflow_backend.client.get_metric_history(run_id, tag)
    assert sorted((m.step, m.value) for m in history) == list(enumerate(values))


def test_params_are_flattened_and_grouped_by_stage(mlflow_backend) -> None:
    run_id = mlflow_backend.new_run()
    tracker = MLflowTracker(run_id, client=mlflow_backend.client)

    tracker.log_params(
        {
            "mean": {"optimizer": {"learning_rate": 1e-3}, "batch_size": 32},
            "variance": {"optimizer": {"learning_rate": 5e-4}},
        }
    )

    assert mlflow_backend.client.get_run(run_id).data.params == {
        "mean/optimizer/learning_rate": "0.001",
        "mean/batch_size": "32",
        "variance/optimizer/learning_rate": "0.0005",
    }


_nested_params = st.recursive(
    st.text(min_size=1, max_size=5),
    lambda children: st.dictionaries(_names, children, min_size=1, max_size=3),
    max_leaves=6,
).filter(lambda tree: isinstance(tree, dict))


@given(params=_nested_params)
def test_unflattening_inverts_the_stage_flattening(params) -> None:
    assert unflatten_parameters(flatten_parameters(params)) == params


def test_a_run_aborted_by_an_exception_is_marked_failed(mlflow_backend) -> None:
    with pytest.raises(RuntimeError), track_mlflow(mlflow_backend.setup) as run:
        run_id = run.info.run_id
        raise RuntimeError("training diverged")

    assert mlflow_backend.client.get_run(run_id).info.status == "FAILED"


def test_backbone_tags_identify_role_seed_data_and_stagereg() -> None:
    assert backbone_tags(seed=3, data_generation_run_id="abc") == {
        "role": "backbone",
        "seed": "3",
        "data_generation_run_id": "abc",
        "stagereg_version": "0.3.0",
    }


def _artifact_paths(client, run_id: str) -> set[str]:
    return {artifact.path for artifact in client.list_artifacts(run_id)}


def test_log_artifact_logs_text_verbatim(mlflow_backend) -> None:
    run_id = mlflow_backend.new_run()
    tracker = MLflowTracker(run_id, client=mlflow_backend.client)

    tracker.log_artifact("training_log.txt", "epoch 0: loss 1.0\n")

    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="training_log.txt"
    )
    assert pathlib.Path(path).read_text() == "epoch 0: loss 1.0\n"


def test_log_artifact_logs_a_mapping_as_json(mlflow_backend) -> None:
    run_id = mlflow_backend.new_run()
    tracker = MLflowTracker(run_id, client=mlflow_backend.client)
    value = {"policies": {"0": "constant", "1": "variable"}, "n": 3}

    tracker.log_artifact("config", value)

    assert mlflow.artifacts.load_dict(f"runs:/{run_id}/config.json") == value


def test_log_artifact_logs_a_figure_as_an_image(mlflow_backend) -> None:
    run_id = mlflow_backend.new_run()
    tracker = MLflowTracker(run_id, client=mlflow_backend.client)
    figure = Figure()
    figure.add_subplot().plot([0, 1], [1, 0])

    tracker.log_artifact("loss_curve", figure)

    assert "loss_curve.png" in _artifact_paths(mlflow_backend.client, run_id)


@pytest.mark.parametrize("value", [1.0, b"bytes", ["a", "list"], None])
def test_log_artifact_rejects_unsupported_values(mlflow_backend, value) -> None:
    run_id = mlflow_backend.new_run()
    tracker = MLflowTracker(run_id, client=mlflow_backend.client)

    with pytest.raises(TypeError):
        tracker.log_artifact("artifact", value)

    assert _artifact_paths(mlflow_backend.client, run_id) == set()


_DTYPES = st.sampled_from([jnp.float32, jnp.int32, jnp.uint32, jnp.bool_])


@st.composite
def _arrays(draw: st.DrawFn) -> jax.Array:
    shape = tuple(draw(st.lists(st.integers(0, 3), max_size=3)))
    dtype = draw(_DTYPES)
    size = int(np.prod(shape))
    values = np.asarray(
        draw(st.lists(st.integers(-1000, 1000), min_size=size, max_size=size))
    )
    if dtype == jnp.float32:
        values = values / 7.0
    elif dtype == jnp.uint32:
        values = np.abs(values)
    return jnp.asarray(values.reshape(shape), dtype=dtype)


_array_trees = st.recursive(
    _arrays(),
    lambda children: st.dictionaries(_names, children, min_size=1, max_size=3),
    max_leaves=5,
)


@st.composite
def _checkpoints(draw: st.DrawFn) -> Checkpoint:
    metric_history = draw(
        st.dictionaries(
            st.sampled_from(["mean/train/loss", "variance/validation/loss"]),
            st.lists(st.floats(allow_nan=False, allow_infinity=False), max_size=3),
            max_size=2,
        )
    )
    state = TrainingState(
        rng_state=jax.random.key(draw(st.integers(0, 2**31 - 1))),
        posterior_state=draw(st.none() | _array_trees),
        lifecycle_state=draw(st.sampled_from(list(StageState))),
        metric_history=metric_history,
    )
    stopping = draw(
        st.none()
        | st.builds(
            lambda patience, bad, best, epoch: EarlyStoppingState(
                metric="loss",
                mode="min",
                min_delta=0.0,
                patience=patience,
                source="validation",
                best_score=best,
                best_epoch=epoch,
                non_improving_epochs=bad,
                stopped=bad > patience,
            ),
            st.integers(0, 60),
            st.integers(0, 60),
            st.floats(allow_nan=False, allow_infinity=False),
            st.integers(0, 100),
        )
    )
    return Checkpoint(
        state=state,
        epoch=draw(st.integers(0, 500)),
        parameters=draw(st.none() | _array_trees),
        optimizer_state=draw(st.none() | _array_trees),
        early_stopping_state=stopping,
        metadata=draw(
            st.dictionaries(_names, st.booleans() | st.integers() | _names, max_size=3)
        ),
    )


@_MLFLOW_SETTINGS
@given(checkpoint=_checkpoints(), key=st.sampled_from(["mean/best", "posterior/best"]))
def test_a_loaded_checkpoint_equals_the_saved_one(
    mlflow_backend, exactly_equal, checkpoint: Checkpoint, key: str
) -> None:
    store = MLflowCheckpointStore(
        mlflow_backend.new_run(), client=mlflow_backend.client
    )

    store.save(key, checkpoint)

    exactly_equal(store.load(key), checkpoint)


def test_a_finalized_backbone_checkpoint_round_trips(
    mlflow_backend, exactly_equal, trained_backbone
) -> None:
    store = MLflowCheckpointStore(
        mlflow_backend.new_run(), client=mlflow_backend.client
    )
    checkpoint = trained_backbone.checkpoints.mean

    store.save("mean/best", checkpoint)

    exactly_equal(store.load("mean/best"), checkpoint)


@pytest.mark.parametrize("activation", ["gelu", "relu", "silu", "tanh"])
def test_a_network_graph_round_trips_for_every_searched_activation(
    mlflow_backend, activation: str
) -> None:
    """Restoring a stage compares graphs, so importable callables keep identity."""
    network = build_mean_net(
        MeanNetworkHyperparameters(activation=activation), jax.random.key(0)
    )
    graph = nnx.graphdef(network)
    store = MLflowCheckpointStore(
        mlflow_backend.new_run(), client=mlflow_backend.client
    )

    store.save(
        "mean/best", Checkpoint(state=TrainingState(), epoch=0, metadata={"g": graph})
    )

    assert store.load("mean/best").metadata["g"] == graph


def test_checkpoint_artifacts_follow_the_checkpoint_key(
    mlflow_backend, trained_backbone
) -> None:
    run_id = mlflow_backend.new_run()
    store = MLflowCheckpointStore(run_id, client=mlflow_backend.client)

    store.save("variance/best", trained_backbone.checkpoints.variance)

    client = mlflow_backend.client
    assert _artifact_paths(client, run_id) == {"checkpoints"}
    assert {a.path for a in client.list_artifacts(run_id, "checkpoints")} == {
        "checkpoints/variance"
    }
    assert {a.path for a in client.list_artifacts(run_id, "checkpoints/variance")} == {
        "checkpoints/variance/best"
    }


def test_a_missing_checkpoint_does_not_exist_and_cannot_be_loaded(
    mlflow_backend,
) -> None:
    store = MLflowCheckpointStore(
        mlflow_backend.new_run(), client=mlflow_backend.client
    )

    assert not store.exists("mean/best")
    with pytest.raises(KeyError):
        store.load("mean/best")


def test_a_saved_checkpoint_exists(mlflow_backend, trained_backbone) -> None:
    store = MLflowCheckpointStore(
        mlflow_backend.new_run(), client=mlflow_backend.client
    )

    store.save("mean/best", trained_backbone.checkpoints.mean)

    assert store.exists("mean/best")
    assert not store.exists("variance/best")


def test_the_mlflow_store_persists_only_finalized_checkpoints(
    mlflow_backend, trained_backbone
) -> None:
    run_id = mlflow_backend.new_run()
    store = mlflow_checkpoint_store(run_id, client=mlflow_backend.client)
    best = dataclasses.replace(trained_backbone.checkpoints.mean, metadata={})

    store.save("mean/best", best)

    assert not MLflowCheckpointStore(run_id, client=mlflow_backend.client).exists(
        "mean/best"
    )
    store.save("mean/best", trained_backbone.checkpoints.mean)
    assert is_finalized(
        MLflowCheckpointStore(run_id, client=mlflow_backend.client).load("mean/best")
    )


# --- Run selection --------------------------------------------------------------


def test_an_explicit_run_id_selects_that_exact_run(make_run) -> None:
    older = make_run(start_time=1, role="backbone", seed=0)
    make_run(start_time=2, role="backbone", seed=0)

    assert (
        resolve_run_id(
            run_id=older,
            criteria=RunCriteria(role="backbone", seed=0),
            experiment_ids=[make_run.experiment_id],
            client=make_run.client,
        )
        == older
    )


def test_an_unknown_run_id_raises(make_run) -> None:
    with pytest.raises(RunSelectionError, match="0123456789abcdef"):
        resolve_run_id(
            run_id="0123456789abcdef",
            criteria=None,
            experiment_ids=[make_run.experiment_id],
            client=make_run.client,
        )


_ROLES = ["backbone", "method", "hpo_study", "hpo_trial"]


@_MLFLOW_SETTINGS
@given(
    run_role=st.sampled_from([*_ROLES, None]),
    expected_role=st.sampled_from(_ROLES),
)
def test_an_explicit_run_id_is_accepted_iff_its_role_matches_the_criteria(
    make_run, run_role: str | None, expected_role: str
) -> None:
    make_run = make_run.fresh()
    tags = {"role": run_role} if run_role is not None else {}
    run_id = make_run(start_time=1, method="hnn", seed=0, **tags)

    def resolve() -> str:
        return resolve_run_id(
            run_id=run_id,
            criteria=RunCriteria(role=expected_role, method="hnn", seed=0),
            experiment_ids=[make_run.experiment_id],
            client=make_run.client,
        )

    if run_role == expected_role:
        assert resolve() == run_id
    else:
        with pytest.raises(RunSelectionError, match=f"expected {expected_role!r}"):
            resolve()


def test_an_explicit_run_id_without_criteria_skips_the_role_check(make_run) -> None:
    run_id = make_run(start_time=1, role="backbone", seed=0)

    assert (
        resolve_run_id(
            run_id=run_id,
            criteria=None,
            experiment_ids=[make_run.experiment_id],
            client=make_run.client,
        )
        == run_id
    )


def _select(make_run, criteria: RunCriteria) -> str:
    return resolve_run_id(
        run_id=None,
        criteria=criteria,
        experiment_ids=[make_run.experiment_id],
        client=make_run.client,
    )


_LATER = 2_000_000


@_MLFLOW_SETTINGS
@given(
    start_times=st.lists(
        st.integers(min_value=1, max_value=1_000_000),
        min_size=1,
        max_size=4,
        unique=True,
    ),
    statuses=st.lists(
        st.sampled_from(["FINISHED", "FAILED", "KILLED"]), min_size=4, max_size=4
    ),
)
def test_tag_selection_returns_the_newest_finished_match(
    make_run, start_times: list[int], statuses: list[str]
) -> None:
    make_run = make_run.fresh()
    statuses = ["FINISHED", *statuses[1:]]  # at least one finished match
    runs = {
        make_run(start_time=t, status=status, role="backbone", seed=3): (t, status)
        for t, status in zip(start_times, statuses, strict=False)
    }
    # Newer runs that differ in one tag never match.
    make_run(start_time=_LATER, role="backbone", seed=4)
    make_run(start_time=_LATER + 1, role="method", method="hnn", seed=3)

    newest_finished = max(
        (t, run_id) for run_id, (t, status) in runs.items() if status == "FINISHED"
    )[1]
    assert _select(make_run, RunCriteria(role="backbone", seed=3)) == newest_finished


def test_tag_selection_without_a_finished_match_raises(make_run) -> None:
    make_run(start_time=1, status="FAILED", role="method", method="q-avi", seed=0)
    make_run(start_time=2, role="method", method="c-avi", seed=0)

    with pytest.raises(RunSelectionError, match="No finished run"):
        _select(make_run, RunCriteria(role="method", method="q-avi", seed=0))


@pytest.mark.parametrize(
    "criteria",
    [
        RunCriteria(role="backbone"),
        RunCriteria(role="method", seed=0),
        RunCriteria(role="method", method="q-avi"),
    ],
)
def test_criteria_that_leave_an_identity_tag_open_are_ambiguous(
    make_run, criteria: RunCriteria
) -> None:
    make_run(start_time=1, role="backbone", seed=0)
    make_run(start_time=2, role="method", method="q-avi", seed=0)

    with pytest.raises(RunSelectionError, match="Ambiguous"):
        _select(make_run, criteria)


def test_newest_matches_with_equal_start_times_are_ambiguous(make_run) -> None:
    make_run(start_time=1, role="backbone", seed=0)
    make_run(start_time=5, role="backbone", seed=0)
    make_run(start_time=5, role="backbone", seed=0)

    with pytest.raises(RunSelectionError, match="Ambiguous"):
        _select(make_run, RunCriteria(role="backbone", seed=0))


def test_an_older_tie_does_not_make_the_newest_match_ambiguous(make_run) -> None:
    make_run(start_time=1, role="backbone", seed=0)
    make_run(start_time=1, role="backbone", seed=0)
    newest = make_run(start_time=2, role="backbone", seed=0)

    assert _select(make_run, RunCriteria(role="backbone", seed=0)) == newest


@pytest.mark.parametrize("role", ["hpo_trial", "hpo_study"])
def test_tag_selection_refuses_hpo_roles(make_run, role: str) -> None:
    make_run(start_time=1, role=role, method="q-avi", seed=0)

    with pytest.raises(ValueError, match=role):
        _select(make_run, RunCriteria(role=role, method="q-avi", seed=0))


def test_resolving_without_run_id_or_criteria_raises(make_run) -> None:
    with pytest.raises(ValueError, match="run ID or tag criteria"):
        resolve_run_id(
            run_id=None,
            criteria=None,
            experiment_ids=[make_run.experiment_id],
            client=make_run.client,
        )


@pytest.mark.parametrize(
    ("selector", "argv_tags"),
    [
        (RunSelector(role="backbone"), ["--seed", "0"]),
        (
            RunSelector(role="backbone", run_id_flag="--backbone-run-id"),
            ["--seed", "0"],
        ),
        (RunSelector(role="method"), ["--method", "q-avi", "--seed", "0"]),
    ],
)
def test_a_selector_resolves_the_command_line_by_run_id_or_tags(
    make_run, selector: RunSelector, argv_tags: list[str]
) -> None:
    role = selector.role
    older = make_run(start_time=1, role=role, method="q-avi", seed=0)
    newest = make_run(start_time=2, role=role, method="q-avi", seed=0)
    parser = argparse.ArgumentParser()
    selector.add_arguments(parser)

    def resolve(argv: list[str]) -> str:
        return selector.resolve(
            parser.parse_args(argv),
            experiment_ids=[make_run.experiment_id],
            client=make_run.client,
        )

    assert resolve([selector.run_id_flag, older]) == older
    assert resolve(argv_tags) == newest


def test_an_hpo_trial_is_never_selected_even_when_newest(make_run) -> None:
    protocol_run = make_run(start_time=1, role="method", method="q-avi", seed=0)
    make_run(start_time=2, role="hpo_trial", method="q-avi", seed=0)

    assert (
        _select(make_run, RunCriteria(role="method", method="q-avi", seed=0))
        == protocol_run
    )


# --- protocol overrides and backbone provenance ----------------------------------


def test_backbone_tags_record_the_studies_the_backbone_was_configured_from() -> None:
    tags = backbone_tags(
        seed=0,
        data_generation_run_id="abc",
        mean_study_run_id="mean-study",
        variance_study_run_id="variance-study",
    )

    assert tags["mean_study_run_id"] == "mean-study"
    assert tags["variance_study_run_id"] == "variance-study"
    assert "variance_study_run_id" not in backbone_tags(
        seed=0, data_generation_run_id="abc", mean_study_run_id="mean-study"
    )


@given(names=st.lists(_names, max_size=4, unique=True))
def test_protocol_override_tags_flag_and_list_exactly_the_overridden_settings(
    names: list[str],
) -> None:
    tags = protocol_override_tags(names)

    if not names:
        assert tags == {}
    else:
        assert tags[PROTOCOL_OVERRIDES_TAG] == "true"
        assert tags[PROTOCOL_OVERRIDES_PARAMS_TAG].split(",") == sorted(names)


def test_tag_selection_skips_runs_with_protocol_overrides(make_run) -> None:
    protocol_run = make_run(start_time=1, role="method", method="bbb", seed=0)
    smoke = make_run(
        start_time=2,
        role="method",
        method="bbb",
        seed=0,
        **protocol_override_tags(["stage/max_epochs"]),
    )

    assert (
        _select(make_run, RunCriteria(role="method", method="bbb", seed=0))
        == protocol_run
    )
    assert (
        _select(
            make_run,
            RunCriteria(role="method", method="bbb", seed=0, include_overridden=True),
        )
        == smoke
    )


def test_a_selector_includes_overridden_runs_only_when_asked(make_run) -> None:
    make_run(start_time=1, role="backbone", seed=0)
    smoke = make_run(
        start_time=2, role="backbone", seed=0, **protocol_override_tags(["x"])
    )
    selector = RunSelector(role="backbone")
    parser = argparse.ArgumentParser()
    selector.add_arguments(parser)

    resolved = selector.resolve(
        parser.parse_args(["--seed", "0", "--include-overridden"]),
        experiment_ids=[make_run.experiment_id],
        client=make_run.client,
    )

    assert resolved == smoke
