from __future__ import annotations

from unittest.mock import Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from qmodem.battery.dispatch import ModelBuildParameters
from qmodem.battery.train import (
    Method,
    QAVITrainHyperparameters,
    TrainHyperparameters,
    load_train_hyperparameters_from_mlflow,
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
