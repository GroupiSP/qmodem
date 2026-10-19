"""MCD's configuration: logged round trip, validation and its posterior network."""

from __future__ import annotations

import jax
import pydantic
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.methods.mcd import MCDConfig
from qmodem.battery.models import MCDropoutNet
from qmodem.battery.train import MeanNetworkHyperparameters, OptimizerHyperparameters

_rates = st.floats(1e-3, 0.999, allow_nan=False)


@given(rate=_rates, learning_rate=st.floats(1e-6, 1.0))
def test_an_mcd_configuration_round_trips_through_its_logged_params(
    round_trip, rate: float, learning_rate: float
) -> None:
    config = MCDConfig(
        dropout_rate=rate,
        optimizer=OptimizerHyperparameters(learning_rate=learning_rate),
    )

    assert round_trip(config) == config


@given(rate=st.one_of(st.floats(-1.0, 0.0), st.floats(1.0, 2.0)))
def test_a_dropout_rate_outside_the_open_unit_interval_is_refused(
    rate: float,
) -> None:
    with pytest.raises(pydantic.ValidationError):
        MCDConfig(dropout_rate=rate)


@settings(max_examples=10, deadline=None)
@given(rate=_rates, kernel_size=st.integers(2, 5), n_filters=st.integers(1, 6))
def test_the_mcd_posterior_network_has_the_configured_rate_and_backbone_sizes(
    rate: float, kernel_size: int, n_filters: int
) -> None:
    spec = MCDConfig(dropout_rate=rate).posterior_spec()
    network = MeanNetworkHyperparameters(kernel_size=kernel_size, n_filters=n_filters)

    net = spec.build_network(network, jax.random.key(0))

    assert isinstance(net, MCDropoutNet)
    assert net.dropout.rate == rate
    assert net.conv.kernel[...].shape == (kernel_size, 2, n_filters)
    assert spec.early_stopping
