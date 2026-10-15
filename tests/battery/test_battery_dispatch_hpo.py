from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.dispatch import (
    ModelBuildParameters,
    build_discriminator,
    build_qavi_model,
)
from qmodem.battery.train import Method, QAVITrainHyperparameters

# ---------------------------------------------------------------------------
# dispatch factory tests
# ---------------------------------------------------------------------------


def test_build_qavi_model_returns_cnn_with_correct_config() -> None:
    from qmodem.battery.models import CNN

    hp = QAVITrainHyperparameters(
        conv_kernel_size=3,
        conv_n_filters=4,
        window_size=10,
        pqc_n_qubits=3,
        pqc_n_layers=1,
    )

    model = build_qavi_model(hp)

    assert isinstance(model, CNN)
    # The GaussianBlock input dimension equals n_filters.
    assert model.gauss.linear_1.in_features == hp.conv_n_filters


def test_build_discriminator_returns_discriminator_with_correct_input_dim() -> None:
    hp = QAVITrainHyperparameters(
        window_size=10,
        discriminator_hidden_size=32,
    )

    discriminator = build_discriminator(hp)

    # input_dim == 2 * window_size + 1
    expected_input_dim = 2 * hp.window_size + 1
    assert discriminator.l1.in_features == expected_input_dim
    assert discriminator.l1.out_features == hp.discriminator_hidden_size


def test_build_discriminator_uses_configured_activation() -> None:
    hp = QAVITrainHyperparameters(discriminator_act_fn="relu")

    discriminator = build_discriminator(hp)

    assert discriminator.act_fn is not None
    assert discriminator.act_fn.__name__ == "relu"


def test_build_model_raises_for_qavi_method() -> None:
    from qmodem.battery.dispatch import build_model

    params = ModelBuildParameters(
        method=Method.Q_AVI,
        conv_n_filters=4,
        conv_kernel_size=3,
        dropout_rate=0.1,
        activation_function="gelu",
        net_init_seed=0,
    )

    with pytest.raises(ValueError, match="QAVI"):
        build_model(params)


@settings(deadline=None)
@given(
    pqc_n_qubits=st.integers(min_value=3, max_value=5),
    pqc_n_layers=st.integers(min_value=1, max_value=3),
    conv_n_filters=st.integers(min_value=2, max_value=8),
)
def test_build_qavi_model_accepts_valid_qubit_and_layer_counts(
    pqc_n_qubits: int,
    pqc_n_layers: int,
    conv_n_filters: int,
) -> None:
    """build_qavi_model must not raise for any valid combination of qubits/layers."""
    hp = QAVITrainHyperparameters(
        pqc_n_qubits=pqc_n_qubits,
        pqc_n_layers=pqc_n_layers,
        conv_n_filters=conv_n_filters,
        conv_kernel_size=3,
        window_size=10,
    )

    # Should not raise.
    model = build_qavi_model(hp)

    assert model.gauss.linear_1.in_features == conv_n_filters
