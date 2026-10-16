"""The AVI method configurations: Q-AVI and C-AVI differ only in their latent map."""

from __future__ import annotations

import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.methods.avi import (
    LATENT_MAP_CLASSICAL_PARAMETERS,
    LATENT_MAP_QUANTUM_PARAMETERS,
    AVIHyperparameters,
    CAVIConfig,
    QAVIConfig,
    classical_latent_map,
    latent_map_parameter_counts,
    quantum_latent_map,
)

_hyperparameters = st.builds(
    AVIHyperparameters,
    adversarial_weight=st.floats(1e-4, 1.0),
    num_draws=st.sampled_from([1, 4, 8, 16]),
    discriminator_steps=st.sampled_from([1, 2, 3, 5]),
    n_qubits=st.integers(3, 6),
    n_layers=st.integers(1, 4),
)


@pytest.mark.parametrize("config_type", [QAVIConfig, CAVIConfig])
@given(hp=_hyperparameters)
def test_an_avi_configuration_round_trips_through_its_logged_params(
    round_trip, config_type, hp: AVIHyperparameters
) -> None:
    config = config_type(avi=hp)

    assert round_trip(config) == config


@settings(max_examples=10, deadline=None)
@given(n_qubits=st.integers(2, 6), n_layers=st.integers(1, 4))
def test_the_logged_latent_map_counts_are_the_built_maps_counts(
    n_qubits: int, n_layers: int
) -> None:
    hp = AVIHyperparameters(n_qubits=n_qubits, n_layers=n_layers)
    counts = latent_map_parameter_counts(hp)

    quantum = quantum_latent_map(hp, nnx.Rngs(0))
    classical = classical_latent_map(hp, nnx.Rngs(0))

    assert counts[LATENT_MAP_QUANTUM_PARAMETERS] == quantum.parameter_count
    assert counts[LATENT_MAP_QUANTUM_PARAMETERS] == 2 * n_layers * n_qubits
    assert counts[LATENT_MAP_CLASSICAL_PARAMETERS] == classical.parameter_count
    assert quantum.width == classical.width == n_qubits


@given(hp=_hyperparameters)
def test_both_avi_methods_log_the_same_latent_map_counts(
    hp: AVIHyperparameters,
) -> None:
    quantum = QAVIConfig(avi=hp).posterior_spec()
    classical = CAVIConfig(avi=hp).posterior_spec()

    assert quantum.method_metrics == classical.method_metrics
    assert quantum.early_stopping and classical.early_stopping
    assert quantum.scaffolding_parameters is not None
