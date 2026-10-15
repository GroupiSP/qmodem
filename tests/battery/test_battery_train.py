from __future__ import annotations

from unittest.mock import Mock

import jax
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from stagereg.core import Checkpoint, is_finalized

from qmodem.battery.dispatch import ModelBuildParameters
from qmodem.battery.train import (
    MeanHyperparameters,
    MeanNetworkHyperparameters,
    Method,
    QAVITrainHyperparameters,
    TrainHyperparameters,
    load_train_hyperparameters_from_mlflow,
    train_backbone,
)


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
