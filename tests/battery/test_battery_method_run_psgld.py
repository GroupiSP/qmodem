"""Seam 1 for pSGLD: train backbone -> train (and restore) a pSGLD method run."""

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
    PSGLDConfig,
    restore_method_run,
    train_method_run,
)
from qmodem.battery.methods.psgld import chain_schedule
from qmodem.battery.posterior import (
    COST_EPOCH_SECONDS,
    COST_SECONDS_PER_EPOCH,
    COST_TRAINABLE_PARAMETERS,
    INFERRED_SUBNETWORK,
    PosteriorStageHyperparameters,
    posterior_loader,
)
from qmodem.battery.train import Method

EPOCHS = 8
TINY_PSGLD = PSGLDConfig(
    burn_in_epochs=3,
    num_samples=6,
    stage=PosteriorStageHyperparameters(max_epochs=EPOCHS),
)


def _backbone_handle(trained_backbone, store_factory) -> BackboneHandle:
    """A handle that sees only what a later process sees: the persisted checkpoints."""
    return BackboneHandle(
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        store=store_factory(trained_backbone.persistent),
    )


def _train(trained_backbone, store_factory, tracker, store) -> MethodRun:
    return train_method_run(
        backbone=_backbone_handle(trained_backbone, store_factory),
        data=trained_backbone.battery.data,
        config=TINY_PSGLD,
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store,
    )


@pytest.fixture(scope="module")
def psgld_run(trained_backbone, tracker_factory, store_factory):
    tracker = tracker_factory()
    persistent = InMemoryCheckpointStore()
    run = _train(trained_backbone, store_factory, tracker, store_factory(persistent))
    return run, tracker, persistent


def _steps_per_epoch(trained_backbone) -> int:
    loader = posterior_loader(
        trained_backbone.battery.data,
        batch_size=TINY_PSGLD.stage.batch_size,
        key=jax.random.key(0),
    )
    return len(loader(split="train", epoch=0))


def test_psgld_is_registered() -> None:
    assert METHOD_CONFIGS[Method.PSGLD] is PSGLDConfig


def test_the_protocol_psgld_config() -> None:
    config = PSGLDConfig()

    assert config.stage.max_epochs == 300
    assert config.burn_in_epochs == 100
    assert config.num_samples == 100
    assert config.posterior_spec().early_stopping is False


def test_a_burn_in_as_long_as_the_run_is_refused() -> None:
    with pytest.raises(ValueError, match="burn_in_epochs"):
        PSGLDConfig(burn_in_epochs=5, stage=PosteriorStageHyperparameters(max_epochs=5))


def test_a_psgld_run_finalizes_its_last_epoch_with_exactly_m_samples(
    psgld_run, trained_backbone
) -> None:
    run, _, persistent = psgld_run
    finalized = persistent.load("posterior/best")
    inputs = trained_backbone.battery.data.posterior_val.features[:7]

    assert is_finalized(finalized)
    assert finalized.epoch == EPOCHS - 1
    assert finalized.early_stopping_state is None
    for leaf in jax.tree.leaves(finalized.state.posterior_state):
        assert leaf.shape[0] == TINY_PSGLD.num_samples
    draws = run.predictive(inputs, jax.random.key(0), num_draws=100).draws
    assert draws.shape == (TINY_PSGLD.num_samples, 7, 1)
    assert not np.allclose(draws[0], draws[-1])


def test_a_psgld_run_validates_only_after_its_first_retained_sample(
    psgld_run, trained_backbone
) -> None:
    _, tracker, _ = psgld_run
    steps = _steps_per_epoch(trained_backbone)
    schedule = chain_schedule(
        epochs=EPOCHS,
        burn_in_epochs=TINY_PSGLD.burn_in_epochs,
        num_samples=TINY_PSGLD.num_samples,
        steps_per_epoch=steps,
    )
    # Epoch e (0-based) ends at step (e + 1) * steps; the first sample is
    # retained at step burn_in + thinning.
    first_validated = (schedule.burn_in + schedule.thinning - 1) // steps

    assert len(tracker.history("posterior/train/loss")) == EPOCHS
    assert len(tracker.history("posterior/validation/crps")) == EPOCHS - first_validated


def test_a_psgld_run_logs_one_validation_baseline_before_its_burn_in(
    psgld_run, one_baseline_before_epoch_0
) -> None:
    _, tracker, _ = psgld_run

    one_baseline_before_epoch_0(tracker)


def test_a_psgld_run_logs_its_time_per_epoch(psgld_run) -> None:
    _, tracker, _ = psgld_run
    epoch_seconds = tracker.history(COST_EPOCH_SECONDS)

    assert len(epoch_seconds) == EPOCHS
    assert all(seconds > 0 for seconds in epoch_seconds)
    assert tracker.history(COST_SECONDS_PER_EPOCH) == pytest.approx(
        [np.mean(epoch_seconds)]
    )


def test_a_psgld_run_counts_the_conv_layer_as_its_trainable_parameters(
    psgld_run, trained_backbone
) -> None:
    _, tracker, _ = psgld_run
    network = trained_backbone.mean_hp.network
    conv_parameters = network.kernel_size * 2 * network.n_filters + network.n_filters

    assert tracker.history(COST_TRAINABLE_PARAMETERS) == [float(conv_parameters)]


def test_a_psgld_run_logs_its_posterior_hyperparameters_with_the_dataset_size(
    psgld_run, trained_backbone
) -> None:
    _, tracker, _ = psgld_run

    assert tracker.params == {
        "posterior": {
            **TINY_PSGLD.posterior_hyperparameters(),
            "dataset_size": len(trained_backbone.battery.data.train),
        }
    }


def test_a_restored_psgld_run_yields_the_trained_runs_draws(
    psgld_run, trained_backbone, store_factory
) -> None:
    run, _, persistent = psgld_run
    restored = restore_method_run(
        backbone=_backbone_handle(trained_backbone, store_factory),
        config=TINY_PSGLD,
        seed=trained_backbone.seed,
        store=store_factory(persistent),
    )
    inputs = trained_backbone.battery.data.posterior_val.features
    key = jax.random.key(5)

    np.testing.assert_array_equal(
        restored.predictive(inputs, key, 100).draws,
        run.predictive(inputs, key, 100).draws,
    )


def test_the_same_seed_gives_identical_finalized_psgld_checkpoints_and_draws(
    psgld_run, trained_backbone, tracker_factory, store_factory, exactly_equal
) -> None:
    run, _, persistent = psgld_run
    again_persistent = InMemoryCheckpointStore()
    again = _train(
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
        again.predictive(inputs, key, 100).draws, run.predictive(inputs, key, 100).draws
    )


@settings(max_examples=10, deadline=None)
@given(start=st.integers(0, 20), size=st.integers(1, 6), seed=st.integers(0, 2**16))
def test_the_psgld_frozen_remainder_keeps_its_trained_mean_values(
    psgld_run, trained_backbone, start: int, size: int, seed: int
) -> None:
    run, _, _ = psgld_run
    mean_checkpoint = trained_backbone.checkpoints.mean.parameters
    mean_net = nnx.merge(mean_checkpoint.graphdef, mean_checkpoint.state)
    trained_conv = nnx.state(mean_net, nnx.All(nnx.Param, INFERRED_SUBNETWORK))
    inputs = trained_backbone.battery.data.posterior_val.features[start : start + size]

    np.testing.assert_array_equal(
        run.posterior.mean_function(trained_conv, inputs, jax.random.key(seed)),
        mean_net(inputs),
    )
