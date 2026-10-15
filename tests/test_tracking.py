"""Tests for qmodem.tracking (seam 2: MLflow adapters on a file-backed backend)."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import pathlib

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from matplotlib.figure import Figure
from stagereg.core import (
    Checkpoint,
    EarlyStoppingState,
    StageState,
    TrainingState,
    is_finalized,
)

from qmodem.tracking import (
    MLflowCheckpointStore,
    MLflowTracker,
    backbone_tags,
    mlflow_checkpoint_store,
    stagereg_version,
    track_mlflow,
)

_MLFLOW_SETTINGS = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

_names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12)


def test_stagereg_version_is_the_installed_distribution_version() -> None:
    assert stagereg_version() == importlib.metadata.version("stagereg") == "0.2.0"


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
        "stagereg_version": "0.2.0",
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
