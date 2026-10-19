"""Reference samples of the simulator: the same for every method run."""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pandas as pd
import pytest
import simbat as sb
from hypothesis import given, settings
from hypothesis import strategies as st
from simbat.simulate import ConstantCurrentDischarge

from qmodem.battery.data_generation import (
    Hyperparameters,
    SplitSpec,
    gaussian_noise,
    reconstruct_true_rul_distribution,
    sim_updater_two_scenarios,
    write_split_histories,
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


# --- Fixed-count, policy-stratified history generation ---------------------------


def _two_policy_config(seed: int) -> tuple[sb.SimulationConfig, np.random.Generator]:
    """A fast two-policy config (low initial SOC) and the RNG it draws from."""
    rng = np.random.default_rng(seed)
    config = sb.SimulationConfig(
        current_policies=[
            ConstantCurrentDischarge(-5.0),
            ConstantCurrentDischarge(-4.0),
        ],
        process_noise_distribution=functools.partial(
            gaussian_noise, rng=rng, noise_std=2e-3
        ),
        measurement_noise_distribution=functools.partial(
            gaussian_noise, rng=rng, noise_std=5e-3
        ),
        dt=5.0,
        soc_0=0.05,
    )
    return config, rng


_split_specs = st.integers(0, 4).flatmap(
    lambda n: st.builds(SplitSpec, n_histories=st.just(n), n_policy_1=st.integers(0, n))
)


def _policy_per_history(histories: pd.DataFrame) -> pd.Series:
    policies = histories.groupby("run_id")["policy_id"].agg(["nunique", "first"])
    assert (policies["nunique"] == 1).all()  # one policy per history
    return policies["first"]


@settings(max_examples=10, deadline=None)
@given(splits=st.lists(_split_specs, min_size=1, max_size=3), seed=st.integers(0, 99))
def test_each_split_has_exactly_its_policy_counts_in_contiguous_run_id_blocks(
    splits: list[SplitSpec], seed: int
) -> None:
    config, rng = _two_policy_config(seed)

    histories = write_split_histories(config, splits, rng)

    policies = _policy_per_history(histories)
    n_total = sum(s.n_histories for s in splits)
    assert list(policies.index) == list(range(n_total))
    start = 0
    for split in splits:
        block = policies.iloc[start : start + split.n_histories]
        assert len(block) == split.n_histories
        assert int((block == 1).sum()) == split.n_policy_1
        assert int((block == 0).sum()) == split.n_histories - split.n_policy_1
        start += split.n_histories


@settings(max_examples=5, deadline=None)
@given(splits=st.lists(_split_specs, min_size=1, max_size=3), seed=st.integers(0, 99))
def test_the_same_seed_gives_identical_histories(
    splits: list[SplitSpec], seed: int
) -> None:
    def generate() -> pd.DataFrame:
        config, rng = _two_policy_config(seed)
        return write_split_histories(config, splits, rng)

    pd.testing.assert_frame_equal(generate(), generate())


@given(n_histories=st.integers(0, 50), excess=st.integers(1, 10))
def test_a_policy_1_count_larger_than_the_split_is_rejected(
    n_histories: int, excess: int
) -> None:
    with pytest.raises(ValueError, match="exceeds"):
        SplitSpec(n_histories=n_histories, n_policy_1=n_histories + excess)


def test_default_splits_follow_the_spec() -> None:
    hp = Hyperparameters()

    assert hp.train_stream_splits() == (
        SplitSpec(n_histories=10, n_policy_1=3),  # train
        SplitSpec(n_histories=20, n_policy_1=6),  # backbone validation set
        SplitSpec(n_histories=20, n_policy_1=6),  # posterior validation set
    )
    assert hp.test_split() == SplitSpec(n_histories=20, n_policy_1=6)


def test_split_sizes_and_policy_counts_are_flat_loggable_params() -> None:
    params = dataclasses.asdict(Hyperparameters())

    assert "n_histories_val" not in params
    for split in ("train", "backbone_val", "posterior_val", "test"):
        assert isinstance(params[f"n_histories_{split}"], int)
        assert isinstance(params[f"n_policy_1_{split}"], int)


def test_a_hyperparameter_policy_1_count_larger_than_its_split_is_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        Hyperparameters(n_histories_posterior_val=4, n_policy_1_posterior_val=5)
