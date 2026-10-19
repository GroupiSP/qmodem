"""HNN: the method without posterior stage."""

from __future__ import annotations

from qmodem.battery.methods.hnn import HNNConfig
from qmodem.battery.methods.registry import METHODS
from qmodem.battery.train import Method


def test_hnn_has_no_posterior_stage_and_no_hyperparameters() -> None:
    config = HNNConfig()

    assert config.posterior_spec() is None
    assert config.posterior_hyperparameters() == {}


def test_hnn_is_registered_without_a_search_space(round_trip) -> None:
    entry = METHODS[Method.HNN]

    assert entry.config is HNNConfig
    assert entry.search_dimensions == ()
    assert round_trip(HNNConfig()) == HNNConfig()
