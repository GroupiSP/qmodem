"""Reference samples of the simulator: the same for every method run."""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import pytest
import simbat as sb
from simbat.simulate import ConstantCurrentDischarge
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.data_generation import (
    gaussian_noise,
    reconstruct_true_rul_distribution,
    sim_updater_two_scenarios,
)


def _config(seed: int) -> sb.SimulationConfig:
    rng = np.random.default_rng(seed)
    return sb.SimulationConfig(
        current_policies=[ConstantCurrentDischarge(-5.0)],
        process_noise_distribution=functools.partial(
            gaussian_noise, rng=rng, noise_std=2e-3
        ),
        measurement_noise_distribution=functools.partial(
            gaussian_noise, rng=rng, noise_std=5e-3
        ),
        dt=5.0,
    )


@pytest.fixture(scope="module")
def test_histories() -> pd.DataFrame:
    time = 5.0 * np.arange(6)
    return pd.DataFrame(
        {
            "run_id": 0,
            "policy_id": 0,
            "time": time,
            "soc": np.linspace(0.3, 0.28, 6),
        }
    )


def _reference(config: sb.SimulationConfig, histories: pd.DataFrame) -> dict:
    return reconstruct_true_rul_distribution(
        test_data=histories,
        simulator_base_config=config,
        n_soc0s=2,
        n_mc_samples=3,
        simulator_updater=sim_updater_two_scenarios,
        event_fn=lambda t0, soc0: False,
    )


@settings(max_examples=5, deadline=None)
@given(seed=st.integers(0, 2**16))
def test_reference_samples_are_the_same_on_every_call_with_one_config(
    test_histories: pd.DataFrame, seed: int
) -> None:
    config = _config(seed)

    first = _reference(config, test_histories)
    again = _reference(config, test_histories)

    assert first.keys() == again.keys() == {"test_case_0"}
    np.testing.assert_array_equal(first["test_case_0"], again["test_case_0"])
    assert np.all(np.isfinite(first["test_case_0"]))
    assert np.ptp(first["test_case_0"]) > 0  # the noise matters
