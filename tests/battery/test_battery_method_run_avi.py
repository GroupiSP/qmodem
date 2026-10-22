"""Seam 1 for the AVI methods: train backbone -> train a method run -> restore it.

Q-AVI and C-AVI differ only in their latent map, so every property is checked for
both.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from stagereg.core import InMemoryCheckpointStore, is_finalized

from qmodem.battery.method_run import (
    METHOD_CONFIGS,
    BackboneHandle,
    MethodRun,
    method_config,
    restore_method_run,
    train_method_run,
)
from qmodem.battery.methods.avi import (
    LATENT_MAP_CLASSICAL_PARAMETERS,
    LATENT_MAP_QUANTUM_PARAMETERS,
    AVIHyperparameters,
    CAVIConfig,
    QAVIConfig,
)
from qmodem.battery.posterior import (
    COST_SCAFFOLDING_PARAMETERS,
    COST_TRAINABLE_PARAMETERS,
    INFERRED_SUBNETWORK,
    PosteriorStageHyperparameters,
    parameter_count,
)
from qmodem.battery.train import Method
from qmodem.tracking import MLflowCheckpointStore

PREDICTIVE_SAMPLE_COUNT = 100

TINY_AVI = AVIHyperparameters(n_qubits=3, n_layers=1, num_draws=2)
TINY_STAGE = PosteriorStageHyperparameters(max_epochs=5, patience=2)
TINY_CONFIGS = {
    Method.Q_AVI: QAVIConfig(avi=TINY_AVI, stage=TINY_STAGE),
    Method.C_AVI: CAVIConfig(avi=TINY_AVI, stage=TINY_STAGE),
}
QUANTUM_PARAMETERS = 6
"""2Lq for q = 3, L = 1."""
CLASSICAL_PARAMETERS = 8
"""H(q + 2) + q for q = 3, h = max(1, round(3/5)) = 1."""
DISCRIMINATOR_PARAMETERS = 5633
"""The 64-64 MLP over a flattened (10, 2) window and its target: 21·64 + 64 + 64·64 + 64
+ 64 + 1."""
LATENT_MAP_PARAMETERS = {
    Method.Q_AVI: QUANTUM_PARAMETERS,
    Method.C_AVI: CLASSICAL_PARAMETERS,
}


def _backbone(trained_backbone, store_factory) -> BackboneHandle:
    return BackboneHandle(
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        store=store_factory(trained_backbone.persistent),
    )


def _train(config, trained_backbone, store_factory, tracker, store) -> MethodRun:
    return train_method_run(
        backbone=_backbone(trained_backbone, store_factory),
        data=trained_backbone.battery.data,
        config=config,
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store,
    )


@pytest.fixture(scope="module", params=sorted(TINY_CONFIGS, key=str))
def method(request: pytest.FixtureRequest) -> Method:
    """Each AVI method in turn."""
    return request.param


@pytest.fixture(scope="module")
def avi_run(method, trained_backbone, tracker_factory, store_factory):
    tracker = tracker_factory()
    persistent = InMemoryCheckpointStore()
    run = _train(
        TINY_CONFIGS[method],
        trained_backbone,
        store_factory,
        tracker,
        store_factory(persistent),
    )
    return run, tracker, persistent


def _merged(snapshot) -> nnx.Module:
    return nnx.merge(snapshot.graphdef, snapshot.state)


def test_an_avi_method_is_registered_and_its_hyperparameters_round_trip(
    method,
) -> None:
    config = TINY_CONFIGS[method]

    assert METHOD_CONFIGS[method] is type(config)
    assert method_config(method, config.posterior_hyperparameters()) == config


def test_a_restored_avi_run_yields_identical_draws(
    method, avi_run, trained_backbone, store_factory
) -> None:
    run, _, persistent = avi_run
    restored = restore_method_run(
        backbone=_backbone(trained_backbone, store_factory),
        config=TINY_CONFIGS[method],
        seed=trained_backbone.seed,
        store=store_factory(persistent),
    )
    inputs = trained_backbone.battery.data.posterior_val.features[:7]
    key = jax.random.key(5)

    expected = run.predictive(inputs, key, num_draws=PREDICTIVE_SAMPLE_COUNT)
    actual = restored.predictive(inputs, key, num_draws=PREDICTIVE_SAMPLE_COUNT)

    assert actual.draws.shape == (PREDICTIVE_SAMPLE_COUNT, 7, 1)
    np.testing.assert_array_equal(actual.draws, expected.draws)
    assert not np.allclose(actual.draws[0], actual.draws[1])


def test_the_same_seed_gives_identical_finalized_avi_checkpoints_and_draws(
    method, avi_run, trained_backbone, tracker_factory, store_factory, exactly_equal
) -> None:
    run, _, persistent = avi_run
    again_persistent = InMemoryCheckpointStore()
    again = _train(
        TINY_CONFIGS[method],
        trained_backbone,
        store_factory,
        tracker_factory(),
        store_factory(again_persistent),
    )
    inputs = trained_backbone.battery.data.posterior_val.features
    key = jax.random.key(11)

    exactly_equal(
        again_persistent.load("posterior/best"), persistent.load("posterior/best")
    )
    np.testing.assert_array_equal(
        again.predictive(inputs, key, PREDICTIVE_SAMPLE_COUNT).draws,
        run.predictive(inputs, key, PREDICTIVE_SAMPLE_COUNT).draws,
    )


def test_the_finalized_avi_checkpoint_holds_the_generator_only(avi_run) -> None:
    run, _, persistent = avi_run
    finalized = persistent.load("posterior/best")

    assert is_finalized(finalized)
    assert set(finalized.state.posterior_state) == {"generator"}
    assert finalized.epoch == run.checkpoint.epoch


def test_an_avi_run_logs_its_diagnostics_every_epoch(avi_run) -> None:
    _, tracker, _ = avi_run
    epochs = len(tracker.history("posterior/train/loss"))

    for name in ("discriminator_loss", "discriminator_accuracy", "weight_spread"):
        history = tracker.history(f"posterior/train/{name}")
        assert len(history) == epochs
        assert all(np.isfinite(history))
    assert all(v > 0 for v in tracker.history("posterior/train/weight_spread"))


def test_an_avi_run_logs_one_validation_baseline_before_its_first_epoch(
    avi_run, one_baseline_before_epoch_0
) -> None:
    _, tracker, _ = avi_run

    one_baseline_before_epoch_0(tracker)


def test_an_avi_run_validates_on_crps_every_epoch_and_diagnoses_its_best_once(
    avi_run, crps_per_epoch_and_diagnostics_once
) -> None:
    _, tracker, persistent = avi_run

    crps_per_epoch_and_diagnostics_once(
        tracker, persistent.load("posterior/best").epoch
    )


def test_an_avi_run_finalizes_its_best_epoch_on_validation_crps(avi_run) -> None:
    _, tracker, persistent = avi_run
    crps = tracker.history("posterior/validation/crps")

    assert persistent.load("posterior/best").epoch == int(np.argmin(crps))


def test_an_avi_run_counts_only_the_generator_as_its_posterior_parameters(
    method, avi_run, trained_backbone
) -> None:
    _, tracker, persistent = avi_run
    network = trained_backbone.mean_hp.network
    conv = network.kernel_size * 2 * network.n_filters + network.n_filters
    q = TINY_AVI.n_qubits
    generator = LATENT_MAP_PARAMETERS[method] + q * conv + conv

    assert (
        parameter_count(persistent.load("posterior/best").state.posterior_state)
        == generator
    )
    assert tracker.history(COST_TRAINABLE_PARAMETERS) == [float(generator)]


def test_an_avi_run_counts_the_discriminator_as_scaffolding(avi_run) -> None:
    _, tracker, _ = avi_run

    assert tracker.history(COST_SCAFFOLDING_PARAMETERS) == [
        float(DISCRIMINATOR_PARAMETERS)
    ]


@settings(max_examples=10, deadline=None)
@given(start=st.integers(0, 20), size=st.integers(1, 6), seed=st.integers(0, 2**16))
def test_the_avi_frozen_remainder_keeps_its_trained_mean_values(
    avi_run, trained_backbone, start: int, size: int, seed: int
) -> None:
    run, _, _ = avi_run
    mean_net = _merged(trained_backbone.checkpoints.mean.parameters)
    trained_conv = nnx.state(mean_net, nnx.All(nnx.Param, INFERRED_SUBNETWORK))
    inputs = trained_backbone.battery.data.posterior_val.features[start : start + size]

    np.testing.assert_array_equal(
        run.posterior.mean_function(trained_conv, inputs, jax.random.key(seed)),
        mean_net(inputs),
    )


@settings(max_examples=10, deadline=None)
@given(start=st.integers(0, 20), size=st.integers(1, 6), seed=st.integers(0, 2**16))
def test_avi_draws_are_consistent_across_input_subsets(
    avi_run, trained_backbone, start: int, size: int, seed: int
) -> None:
    run, _, _ = avi_run
    inputs = trained_backbone.battery.data.posterior_val.features
    key = jax.random.key(seed)

    whole = run.posterior.sample_means(inputs, key, 8)
    subset = run.posterior.sample_means(inputs[start : start + size], key, 8)

    np.testing.assert_allclose(
        subset, whole[:, start : start + size], rtol=1e-5, atol=1e-6
    )


def test_a_finalized_avi_checkpoint_round_trips_through_mlflow(
    avi_run, mlflow_backend, exactly_equal
) -> None:
    _, _, persistent = avi_run
    checkpoint = persistent.load("posterior/best")
    store = MLflowCheckpointStore(
        mlflow_backend.new_run(), client=mlflow_backend.client
    )

    store.save("posterior/best", checkpoint)

    exactly_equal(store.load("posterior/best"), checkpoint)


def test_an_avi_run_logs_both_latent_maps_parameter_counts(avi_run) -> None:
    _, tracker, _ = avi_run

    assert tracker.history(LATENT_MAP_QUANTUM_PARAMETERS) == [QUANTUM_PARAMETERS]
    assert tracker.history(LATENT_MAP_CLASSICAL_PARAMETERS) == [CLASSICAL_PARAMETERS]
