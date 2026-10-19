"""BBB's configuration: logged round trip, validation and its posterior plug-in."""

from __future__ import annotations

import jax
import pydantic
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qmodem.battery.methods.bbb import BBBConfig
from qmodem.battery.posterior import PosteriorStageHyperparameters
from qmodem.battery.train import OptimizerHyperparameters

_positive = st.floats(1e-6, 1.0, allow_nan=False)


@given(
    learning_rate=_positive,
    initial_std=_positive,
    max_epochs=st.integers(1, 500),
    patience=st.integers(0, 100),
)
def test_a_bbb_configuration_round_trips_through_its_logged_params(
    round_trip,
    learning_rate: float,
    initial_std: float,
    max_epochs: int,
    patience: int,
) -> None:
    config = BBBConfig(
        optimizer=OptimizerHyperparameters(learning_rate=learning_rate),
        initial_std=initial_std,
        stage=PosteriorStageHyperparameters(max_epochs=max_epochs, patience=patience),
    )

    assert round_trip(config) == config


@given(initial_std=st.floats(-1.0, 0.0))
def test_a_non_positive_initial_std_is_refused(initial_std: float) -> None:
    with pytest.raises(pydantic.ValidationError):
        BBBConfig(initial_std=initial_std)


def test_bbb_early_stops_and_builds_a_fresh_method_each_time() -> None:
    spec = BBBConfig().posterior_spec()
    key = jax.random.key(0)

    assert spec.early_stopping
    assert spec.build_network is None
    assert spec.scaffolding_parameters is None
    assert spec.build_method(key) is not spec.build_method(key)
