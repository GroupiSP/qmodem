"""Seam 1: the run pipeline, train backbone -> train method run."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from flax import nnx
from stagereg.core import InMemoryCheckpointStore

from qmodem.battery.method_run import (
    METHOD_CONFIGS,
    method_key,
    BackboneHandle,
    HNNConfig,
    restore_method_run,
    train_method_run,
)
from qmodem.battery.train import Method, backbone_keys

PREDICTIVE_SAMPLE_COUNT = 100


def _fresh_backbone_handle(trained_backbone, store_factory) -> BackboneHandle:
    """A handle that sees only what a later process sees: the persisted checkpoints."""
    return BackboneHandle(
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        store=store_factory(trained_backbone.persistent),
    )


def _merged(snapshot) -> nnx.Module:
    return nnx.merge(snapshot.graphdef, snapshot.state)


@pytest.fixture(scope="module")
def hnn_run(trained_backbone, tracker_factory, store_factory):
    tracker = tracker_factory()
    persistent = InMemoryCheckpointStore()
    run = train_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        data=trained_backbone.battery.data,
        config=HNNConfig(),
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store_factory(persistent),
    )
    return run, tracker, persistent


def test_an_hnn_run_yields_100_predictive_draws_of_the_backbone_mean_per_window(
    hnn_run, trained_backbone
) -> None:
    run, _, _ = hnn_run
    inputs = trained_backbone.battery.data.val.features[:7]

    predictive = run.predictive(
        inputs, jax.random.key(1), num_draws=PREDICTIVE_SAMPLE_COUNT
    )

    mean = _merged(trained_backbone.checkpoints.mean.parameters)(inputs)
    variance = _merged(trained_backbone.checkpoints.variance.parameters)(inputs)
    assert predictive.draws.shape == (PREDICTIVE_SAMPLE_COUNT, 7, 1)
    for draw in predictive.draws:
        np.testing.assert_array_equal(draw, mean)
    np.testing.assert_array_equal(predictive.aleatoric_variance, variance.mean())


def test_an_hnn_run_trains_and_saves_nothing(hnn_run) -> None:
    run, tracker, persistent = hnn_run

    assert run.checkpoint is None
    assert tracker.metrics == []
    assert not persistent.exists("posterior/best")


def test_a_restored_hnn_run_predicts_like_the_trained_one(
    hnn_run, trained_backbone, store_factory
) -> None:
    run, _, persistent = hnn_run
    restored = restore_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        config=HNNConfig(),
        seed=trained_backbone.seed,
        store=store_factory(persistent),
    )
    inputs = trained_backbone.battery.data.val.features[:5]
    key = jax.random.key(3)

    expected = run.predictive(inputs, key, num_draws=4)
    actual = restored.predictive(inputs, key, num_draws=4)
    np.testing.assert_array_equal(actual.draws, expected.draws)
    np.testing.assert_array_equal(
        actual.aleatoric_variance, expected.aleatoric_variance
    )


def test_every_registered_config_describes_its_own_method() -> None:
    assert Method.HNN in METHOD_CONFIGS
    for method, config_type in METHOD_CONFIGS.items():
        assert config_type().method == method


def _key_bits(key: jax.Array) -> tuple[int, ...]:
    return tuple(np.asarray(jax.random.key_data(key)).ravel().tolist())


@given(seed=st.integers(0, 2**31 - 1))
def test_method_keys_are_reproducible_and_distinct_from_each_other_and_the_backbone(
    seed: int,
) -> None:
    keys = {method: _key_bits(method_key(seed, method)) for method in Method}
    backbone = backbone_keys(seed)

    assert keys == {method: _key_bits(method_key(seed, method)) for method in Method}
    assert len(set(keys.values())) == len(Method)
    assert not set(keys.values()) & {
        _key_bits(k)
        for k in (backbone.mean_init, backbone.variance_init, backbone.loader)
    }
