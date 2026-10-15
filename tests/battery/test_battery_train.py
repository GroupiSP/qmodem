from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from typing import Any
from unittest.mock import Mock

import jax
import numpy as np
import pytest
import sklearn.preprocessing as skpp
from flax import nnx
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from stagereg.core import Checkpoint, is_finalized

from qmodem.battery.data_processing import FEATURES, WINDOW_SIZE
from qmodem.battery.dispatch import ModelBuildParameters
from qmodem.battery.train import (
    MeanHyperparameters,
    MeanNetworkHyperparameters,
    Method,
    QAVITrainHyperparameters,
    TrackedBackbone,
    TrainHyperparameters,
    load_train_hyperparameters_from_mlflow,
    track_backbone_run,
    train_backbone,
)
from qmodem.tracking import MLflowCheckpointStore, ScalerArtifact


def test_train_hyperparameters_defaults_are_valid() -> None:
    hp = TrainHyperparameters()

    assert hp.method == Method.HNN


def test_qavi_train_hyperparameters_defaults_are_valid() -> None:
    hp = QAVITrainHyperparameters()

    assert hp.method == Method.QAVI


@pytest.mark.parametrize("method", ["mcd", "bnn"])
def test_train_hyperparameters_accepts_standard_methods(method: str) -> None:
    hp = TrainHyperparameters(method=method)

    assert hp.method == method


def test_train_hyperparameters_rejects_qavi_method() -> None:
    with pytest.raises(ValidationError):
        TrainHyperparameters(method="qavi")


def test_qavi_train_hyperparameters_rejects_standard_method() -> None:
    with pytest.raises(ValidationError):
        QAVITrainHyperparameters(method="hnn")


def test_train_hyperparameters_is_frozen() -> None:
    hp = TrainHyperparameters()

    with pytest.raises(ValidationError):
        hp.window_size = 50


def test_load_train_hyperparameters_from_mlflow_round_trips_standard_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = TrainHyperparameters(method="bnn", window_size=15, learning_rate=0.005)
    raw_params = {key: str(value) for key, value in original.model_dump().items()}
    monkeypatch.setattr(
        "qmodem.battery.train.get_run_parameters",
        Mock(return_value=raw_params),
    )

    loaded = load_train_hyperparameters_from_mlflow(
        TrainHyperparameters, run_id="fake-run", backend_store="sqlite:///fake.db"
    )

    assert loaded == original


def test_load_train_hyperparameters_from_mlflow_round_trips_qavi_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = QAVITrainHyperparameters(pqc_n_qubits=8, pqc_n_layers=3)
    raw_params = {key: str(value) for key, value in original.model_dump().items()}
    monkeypatch.setattr(
        "qmodem.battery.train.get_run_parameters",
        Mock(return_value=raw_params),
    )

    loaded = load_train_hyperparameters_from_mlflow(
        QAVITrainHyperparameters, run_id="fake-run", backend_store="sqlite:///fake.db"
    )

    assert loaded == original


def test_model_build_parameters_derived_from_train_hyperparameters() -> None:
    train_hp = TrainHyperparameters(
        method="mcd",
        conv_kernel_size=7,
        conv_n_filters=16,
        dropout_rate=0.25,
        activation_function="relu",
        net_init_seed=42,
    )

    build_params = ModelBuildParameters.model_validate(train_hp, from_attributes=True)

    assert build_params.method == Method.MCD
    assert build_params.conv_kernel_size == 7
    assert build_params.conv_n_filters == 16
    assert build_params.dropout_rate == pytest.approx(0.25)
    assert build_params.activation_function == "relu"
    assert build_params.net_init_seed == 42


@given(
    method=st.sampled_from(["hnn", "mcd", "bnn"]),
    conv_kernel_size=st.integers(min_value=1, max_value=15),
    conv_n_filters=st.integers(min_value=1, max_value=64),
    dropout_rate=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    normalize_rul=st.booleans(),
)
def test_train_hyperparameters_string_round_trip_is_lossless(
    method: str,
    conv_kernel_size: int,
    conv_n_filters: int,
    dropout_rate: float,
    normalize_rul: bool,
) -> None:
    """Any valid TrainHyperparameters instance must survive being stringified (as MLflow
    does when logging/retrieving params) and re-validated."""
    original = TrainHyperparameters(
        method=method,
        conv_kernel_size=conv_kernel_size,
        conv_n_filters=conv_n_filters,
        dropout_rate=dropout_rate,
        normalize_rul=normalize_rul,
    )

    raw_params = {key: str(value) for key, value in original.model_dump().items()}
    reloaded = TrainHyperparameters.model_validate(raw_params)

    assert reloaded == original


# --- Backbone run ---------------------------------------------------------


def _leaves(checkpoint: Checkpoint) -> list[np.ndarray]:
    """Every array a checkpoint carries: weights, optimizer state and RNG key."""
    trees = (
        checkpoint.parameters.state,
        checkpoint.optimizer_state.state,
        jax.random.key_data(checkpoint.state.rng_state),
    )
    return [np.asarray(leaf) for leaf in jax.tree.leaves(trees)]


def _assert_bit_identical(first: Checkpoint, second: Checkpoint) -> None:
    first_leaves, second_leaves = _leaves(first), _leaves(second)
    assert len(first_leaves) == len(second_leaves)
    for a, b in zip(first_leaves, second_leaves, strict=True):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def test_mean_hyperparameters_default_to_the_pre_study_backbone() -> None:
    network = MeanHyperparameters().network

    assert (network.kernel_size, network.n_filters, network.activation) == (
        3,
        29,
        "gelu",
    )


def test_mean_hyperparameters_are_frozen() -> None:
    hp = MeanHyperparameters()

    with pytest.raises(ValidationError):
        hp.batch_size = 64


def test_mean_network_rejects_unknown_activation() -> None:
    with pytest.raises(ValidationError):
        MeanNetworkHyperparameters(activation="softmax")


def test_backbone_returns_finalized_mean_and_variance_checkpoints(
    trained_backbone,
) -> None:
    checkpoints = trained_backbone.checkpoints

    assert is_finalized(checkpoints.mean)
    assert is_finalized(checkpoints.variance)
    assert checkpoints.mean.metadata["stage"] == "mean"
    assert checkpoints.variance.metadata["stage"] == "variance"


def test_backbone_persists_only_its_finalized_checkpoints(trained_backbone) -> None:
    persistent = trained_backbone.persistent

    for key in ("mean/best", "variance/best"):
        assert is_finalized(persistent.load(key))
    assert not persistent.exists("posterior/best")


def test_backbone_logs_both_stages_losses(trained_backbone) -> None:
    assert {
        "mean/train/loss",
        "mean/validation/loss",
        "variance/train/loss",
        "variance/validation/loss",
    } <= trained_backbone.tracker.tags


def test_backbone_logs_hyperparameters_grouped_by_stage(trained_backbone) -> None:
    params = trained_backbone.tracker.params

    assert params["mean"] == trained_backbone.mean_hp.model_dump()
    assert params["variance"] == trained_backbone.variance_hp.model_dump()


def test_same_seed_gives_bit_identical_backbone_checkpoints(
    trained_backbone, tracker_factory, store_factory
) -> None:
    again = train_backbone(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        tracker=tracker_factory(),
        store=store_factory(),
    )

    _assert_bit_identical(again.mean, trained_backbone.checkpoints.mean)
    _assert_bit_identical(again.variance, trained_backbone.checkpoints.variance)


def test_another_seed_gives_another_backbone(
    trained_backbone, tracker_factory, store_factory
) -> None:
    other = train_backbone(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed + 1,
        tracker=tracker_factory(),
        store=store_factory(),
    )

    first_weights = jax.tree.leaves(other.mean.parameters.state)
    second_weights = jax.tree.leaves(trained_backbone.checkpoints.mean.parameters.state)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(first_weights, second_weights, strict=True)
    )


@pytest.mark.parametrize("stage", ["mean", "variance"])
def test_backbone_training_loss_decreases(trained_backbone, stage: str) -> None:
    losses = trained_backbone.tracker.history(f"{stage}/train/loss")

    assert losses[-1] < losses[0]


# --- Backbone run tracked in MLflow (seam 2) --------------------------------------

_DATA_GENERATION_RUN_ID = "data-generation-run"


@dataclasses.dataclass(frozen=True)
class _MLflowBackbone:
    run: TrackedBackbone
    scalers: tuple[ScalerArtifact, ...]
    backend: Any
    reference: Any


@pytest.fixture(scope="module")
def mlflow_backbone(mlflow_backend, trained_backbone) -> _MLflowBackbone:
    """A tiny backbone run tracked in the file-backed MLflow backend."""
    rng = np.random.default_rng(0)
    x_scaler = skpp.StandardScaler().fit(rng.normal(3.0, 2.0, (50, len(FEATURES))))
    y_scaler = skpp.MinMaxScaler().fit(rng.uniform(0.0, 900.0, (50, 1)))
    scalers = (
        ScalerArtifact(
            name="x_scaler",
            scaler=x_scaler,
            input_example=np.ones((WINDOW_SIZE, len(FEATURES))),
        ),
        ScalerArtifact(
            name="y_scaler",
            scaler=y_scaler,
            input_example=np.ones((1, 1)),
            predict_method="inverse_transform",
        ),
    )
    run = track_backbone_run(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        data_generation_run_id=_DATA_GENERATION_RUN_ID,
        scalers=scalers,
        setup=mlflow_backend.setup,
    )
    return _MLflowBackbone(
        run=run, scalers=scalers, backend=mlflow_backend, reference=trained_backbone
    )


def test_the_backbone_run_carries_its_four_tags(mlflow_backbone) -> None:
    run = mlflow_backbone.backend.client.get_run(mlflow_backbone.run.run_id)

    assert {
        key: run.data.tags.get(key)
        for key in ("role", "seed", "data_generation_run_id", "stagereg_version")
    } == {
        "role": "backbone",
        "seed": str(mlflow_backbone.reference.seed),
        "data_generation_run_id": _DATA_GENERATION_RUN_ID,
        "stagereg_version": "0.2.0",
    }
    assert run.info.status == "FINISHED"


def test_only_the_finalized_backbone_checkpoints_reach_mlflow(
    mlflow_backbone, exactly_equal
) -> None:
    client = mlflow_backbone.backend.client
    run_id = mlflow_backbone.run.run_id
    store = MLflowCheckpointStore(run_id, client=client)

    stage_dirs = {a.path for a in client.list_artifacts(run_id, "checkpoints")}
    assert stage_dirs == {"checkpoints/mean", "checkpoints/variance"}
    for stage in ("mean", "variance"):
        assert {
            a.path for a in client.list_artifacts(run_id, f"checkpoints/{stage}")
        } == {f"checkpoints/{stage}/best"}
    exactly_equal(store.load("mean/best"), mlflow_backbone.run.checkpoints.mean)
    exactly_equal(store.load("variance/best"), mlflow_backbone.run.checkpoints.variance)
    assert is_finalized(store.load("mean/best"))
    assert is_finalized(store.load("variance/best"))


@pytest.mark.parametrize("stage", ["mean", "variance"])
def test_backbone_losses_arrive_with_the_stage_epoch_as_step(
    mlflow_backbone, stage: str
) -> None:
    history = mlflow_backbone.backend.client.get_metric_history(
        mlflow_backbone.run.run_id, f"{stage}/train/loss"
    )

    steps = sorted(metric.step for metric in history)
    assert steps == list(range(len(steps)))
    assert len(steps) >= 2


def test_backbone_hyperparameters_arrive_flattened_by_stage(mlflow_backbone) -> None:
    params = mlflow_backbone.backend.client.get_run(
        mlflow_backbone.run.run_id
    ).data.params
    mean_hp = mlflow_backbone.reference.mean_hp
    variance_hp = mlflow_backbone.reference.variance_hp

    assert params["mean/optimizer/learning_rate"] == str(
        mean_hp.optimizer.learning_rate
    )
    assert params["mean/network/n_filters"] == str(mean_hp.network.n_filters)
    assert params["variance/schedule/patience"] == str(variance_hp.schedule.patience)


def test_the_tracked_backbone_matches_the_untracked_one(
    mlflow_backbone, exactly_equal
) -> None:
    exactly_equal(
        mlflow_backbone.run.checkpoints.mean.parameters,
        mlflow_backbone.reference.checkpoints.mean.parameters,
    )


_FRESH_PROCESS_RESTORE = """
import sys

import jax.numpy as jnp
import numpy as np
from flax import nnx

from qmodem.tracking import MLflowCheckpointStore, load_scaler

run_id, out = sys.argv[1], sys.argv[2]
store = MLflowCheckpointStore(run_id)
inputs = jnp.asarray(np.load(out)["inputs"])
mean = store.load("mean/best").parameters
variance = store.load("variance/best").parameters
np.savez(
    out,
    mean=np.asarray(nnx.merge(mean.graphdef, mean.state)(inputs)),
    variance=np.asarray(nnx.merge(variance.graphdef, variance.state)(inputs).mean()),
    x_scaled=load_scaler(run_id, "x_scaler").transform(np.load(out)["features"]),
    y_rul=load_scaler(run_id, "y_scaler").inverse_transform(np.load(out)["targets"]),
)
"""


def test_backbone_checkpoints_and_scalers_restore_in_a_fresh_process(
    mlflow_backbone, tmp_path
) -> None:
    inputs = np.asarray(mlflow_backbone.reference.battery.data.val.features[:5])
    features = np.linspace(0.5, 1.5, 2 * WINDOW_SIZE).reshape(WINDOW_SIZE, 2)
    targets = np.linspace(0.0, 1.0, 4).reshape(4, 1)
    out = tmp_path / "restored.npz"
    np.savez(out, inputs=inputs, features=features, targets=targets)

    subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_RESTORE, mlflow_backbone.run.run_id, out],
        check=True,
        env={
            **os.environ,
            "MLFLOW_TRACKING_URI": mlflow_backbone.backend.setup.backend_store,
        },
    )

    restored = np.load(out)
    checkpoints = mlflow_backbone.run.checkpoints
    mean = nnx.merge(
        checkpoints.mean.parameters.graphdef, checkpoints.mean.parameters.state
    )
    variance = nnx.merge(
        checkpoints.variance.parameters.graphdef, checkpoints.variance.parameters.state
    )
    np.testing.assert_array_equal(restored["mean"], np.asarray(mean(inputs)))
    np.testing.assert_array_equal(
        restored["variance"], np.asarray(variance(inputs).mean())
    )
    x_scaler, y_scaler = (s.scaler for s in mlflow_backbone.scalers)
    np.testing.assert_allclose(restored["x_scaled"], x_scaler.transform(features))
    np.testing.assert_allclose(restored["y_rul"], y_scaler.inverse_transform(targets))
