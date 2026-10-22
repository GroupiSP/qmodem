"""Seam 1: the run pipeline, train backbone -> train method run."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from stagereg.core import InMemoryCheckpointStore, is_finalized
from stagereg.jax import MetricSuite

from qmodem.battery import method_run
from qmodem.battery.data_processing import ValidationSet
from qmodem.battery.method_run import (
    METHOD_CONFIGS,
    BackboneHandle,
    BBBConfig,
    HNNConfig,
    MCDConfig,
    MethodRun,
    method_key,
    posterior_overrides,
    restore_method_run,
    train_method_run,
)
from qmodem.battery.methods.mcd import build_mcd_net
from qmodem.battery.posterior import (
    COST_DRAWS_PER_SECOND,
    COST_TRAINABLE_PARAMETERS,
    COST_WALL_CLOCK,
    DRAWS_PER_SECOND_WINDOWS,
    INFERRED_SUBNETWORK,
    VAL_CRPS_BASELINE,
    VALIDATION_METRIC,
    VALIDATION_TAG_PREFIX,
    PosteriorKeys,
    PosteriorStageHyperparameters,
    diagnostic_metrics,
    posterior_loader,
    score_validation,
    validation_metrics,
)
from qmodem.battery.train import Method, OptimizerHyperparameters, backbone_keys

PREDICTIVE_SAMPLE_COUNT = 100
CRPS = f"{VALIDATION_TAG_PREFIX}{VALIDATION_METRIC}"
"""The per-epoch posterior validation criterion's tag."""
COST_TAGS = {COST_WALL_CLOCK, COST_TRAINABLE_PARAMETERS, COST_DRAWS_PER_SECOND}


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
    inputs = trained_backbone.battery.data.posterior_val.features[:7]

    predictive = run.predictive(
        inputs, jax.random.key(1), num_draws=PREDICTIVE_SAMPLE_COUNT
    )

    mean = _merged(trained_backbone.checkpoints.mean.parameters)(inputs)
    variance = _merged(trained_backbone.checkpoints.variance.parameters)(inputs)
    assert predictive.draws.shape == (PREDICTIVE_SAMPLE_COUNT, 7, 1)
    for draw in predictive.draws:
        np.testing.assert_array_equal(draw, mean)
    np.testing.assert_array_equal(predictive.aleatoric_variance, variance.mean())


def test_an_hnn_run_trains_and_saves_nothing_and_logs_only_its_cost(hnn_run) -> None:
    run, tracker, persistent = hnn_run

    assert run.checkpoint is None
    assert tracker.tags == COST_TAGS
    assert tracker.history(COST_WALL_CLOCK) == [0.0]
    assert tracker.history(COST_TRAINABLE_PARAMETERS) == [0.0]
    assert tracker.history(COST_DRAWS_PER_SECOND)[0] > 0
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
    inputs = trained_backbone.battery.data.posterior_val.features[:5]
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


# --- BBB: the first method with a posterior stage --------------------------------

TINY_BBB = BBBConfig(stage=PosteriorStageHyperparameters(max_epochs=8, patience=2))


def _train_bbb(trained_backbone, store_factory, tracker, store) -> MethodRun:
    return train_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        data=trained_backbone.battery.data,
        config=TINY_BBB,
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store,
    )


@pytest.fixture(scope="module")
def bbb_run(trained_backbone, tracker_factory, store_factory):
    tracker = tracker_factory()
    persistent = InMemoryCheckpointStore()
    run = _train_bbb(
        trained_backbone, store_factory, tracker, store_factory(persistent)
    )
    return run, tracker, persistent


def _restore_bbb(trained_backbone, store_factory, persistent) -> MethodRun:
    return restore_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        config=TINY_BBB,
        seed=trained_backbone.seed,
        store=store_factory(persistent),
    )


def test_a_restored_bbb_run_yields_the_trained_runs_predictive_samples(
    bbb_run, trained_backbone, store_factory
) -> None:
    run, _, persistent = bbb_run
    restored = _restore_bbb(trained_backbone, store_factory, persistent)
    inputs = trained_backbone.battery.data.posterior_val.features[:7]
    key = jax.random.key(5)

    expected = run.predictive(inputs, key, num_draws=PREDICTIVE_SAMPLE_COUNT)
    actual = restored.predictive(inputs, key, num_draws=PREDICTIVE_SAMPLE_COUNT)

    assert actual.draws.shape == (PREDICTIVE_SAMPLE_COUNT, 7, 1)
    assert actual.aleatoric_variance.shape == (7, 1)
    np.testing.assert_array_equal(actual.draws, expected.draws)
    assert not np.allclose(actual.draws[0], actual.draws[1])


def test_the_same_seed_gives_identical_finalized_bbb_checkpoints_and_draws(
    bbb_run, trained_backbone, tracker_factory, store_factory, exactly_equal
) -> None:
    run, _, persistent = bbb_run
    again_persistent = InMemoryCheckpointStore()
    again = _train_bbb(
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


def test_a_bbb_run_finalizes_its_best_epoch_on_validation_crps(bbb_run) -> None:
    run, tracker, persistent = bbb_run
    crps = tracker.history(CRPS)
    finalized = persistent.load("posterior/best")

    assert is_finalized(finalized)
    assert finalized.epoch == int(np.argmin(crps))
    assert finalized.early_stopping_state.best_score == min(crps)
    assert run.checkpoint.epoch == finalized.epoch


def test_a_bbb_run_logs_its_cost(bbb_run, trained_backbone) -> None:
    _, tracker, _ = bbb_run
    network = trained_backbone.mean_hp.network
    conv_parameters = network.kernel_size * 2 * network.n_filters + network.n_filters

    assert COST_TAGS <= tracker.tags
    assert tracker.history(COST_WALL_CLOCK)[0] > 0
    assert tracker.history(COST_TRAINABLE_PARAMETERS) == [2.0 * conv_parameters]
    assert tracker.history(COST_DRAWS_PER_SECOND)[0] > 0


def test_a_bbb_run_logs_only_its_posterior_hyperparameters_with_the_dataset_size(
    bbb_run, trained_backbone
) -> None:
    _, tracker, _ = bbb_run

    assert set(tracker.params) == {"posterior"}
    assert tracker.params["posterior"] == {
        **TINY_BBB.posterior_hyperparameters(),
        "dataset_size": len(trained_backbone.battery.data.train),
    }


def test_a_bbb_run_validates_on_crps_every_epoch_and_diagnoses_its_best_once(
    bbb_run, crps_per_epoch_and_diagnostics_once
) -> None:
    _, tracker, persistent = bbb_run
    epochs = len(tracker.history("posterior/train/loss"))

    assert len(tracker.history(CRPS)) == epochs
    crps_per_epoch_and_diagnostics_once(
        tracker, persistent.load("posterior/best").epoch
    )


def _full_suite() -> MetricSuite:
    """Compose the pre-#139 per-epoch suite: the criterion plus the diagnostics.

    Returns:
        ``validation_metrics()`` and ``diagnostic_metrics()`` merged into one
        suite, so one pass scores all of them on the same predictive samples.

    Raises:
        ValueError: If the two suites' predictors or sample counts differ.
    """
    criterion, diagnostics = validation_metrics(), diagnostic_metrics()
    if criterion.predictive_sample_count != diagnostics.predictive_sample_count:
        raise ValueError("the suites score different predictive sample counts.")
    if type(criterion.predictor) is not type(diagnostics.predictor):
        raise ValueError("the suites use different predictors.")
    return dataclasses.replace(criterion, epoch=criterion.epoch + diagnostics.epoch)


@pytest.mark.parametrize("fixture", ["bbb_run", "mcd_run"])
def test_the_diagnostics_and_best_crps_are_the_full_suites_scores_of_the_best_epoch(
    fixture, request, trained_backbone
) -> None:
    # The full suite on the finalized posterior, under the stage's validation key,
    # reproduces both the logged diagnostics and the best epoch's logged CRPS.
    run, tracker, persistent = request.getfixturevalue(fixture)
    config = TINY_BBB if fixture == "bbb_run" else TINY_MCD
    keys = PosteriorKeys.of(run.key)
    full = score_validation(
        lambda inputs, key: run.predictive(inputs, key, PREDICTIVE_SAMPLE_COUNT),
        posterior_loader(
            trained_backbone.battery.data,
            batch_size=config.stage.batch_size,
            key=keys.loader,
        ),
        keys.validation,
        _full_suite(),
    )
    best = persistent.load("posterior/best").epoch
    crps_steps = tracker.steps(CRPS)
    best_crps = tracker.history(CRPS)[crps_steps.index(best)]

    assert full["crps"] == pytest.approx(best_crps, rel=1e-5)
    for name in ("nll", "rmse", "coverage_95"):
        assert tracker.history(f"{VALIDATION_TAG_PREFIX}{name}") == [
            pytest.approx(full[name], rel=1e-5)
        ]


@settings(max_examples=10, deadline=None)
@given(start=st.integers(0, 20), size=st.integers(1, 6), seed=st.integers(0, 2**16))
def test_the_bbb_frozen_remainder_keeps_its_trained_mean_values(
    bbb_run, trained_backbone, start: int, size: int, seed: int
) -> None:
    run, _, _ = bbb_run
    mean_net = _merged(trained_backbone.checkpoints.mean.parameters)
    trained_conv = nnx.state(mean_net, nnx.All(nnx.Param, INFERRED_SUBNETWORK))
    inputs = trained_backbone.battery.data.posterior_val.features[start : start + size]

    # With the conv layer at its trained-mean values, every leaf of the posterior
    # network equals the trained mean's, so it is the trained mean network.
    np.testing.assert_array_equal(
        run.posterior.mean_function(trained_conv, inputs, jax.random.key(seed)),
        mean_net(inputs),
    )


def test_the_bbb_posterior_loss_decreases(bbb_run) -> None:
    _, tracker, _ = bbb_run
    losses = tracker.history("posterior/train/loss")

    assert losses[-1] < losses[0]


# --- MCD: input dropout before the conv, fine-tuned by MCDropout ------------------

TINY_MCD = MCDConfig(stage=PosteriorStageHyperparameters(max_epochs=8, patience=2))


def _train_mcd(trained_backbone, store_factory, tracker, store) -> MethodRun:
    return train_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        data=trained_backbone.battery.data,
        config=TINY_MCD,
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store,
    )


@pytest.fixture(scope="module")
def mcd_run(trained_backbone, tracker_factory, store_factory):
    tracker = tracker_factory()
    persistent = InMemoryCheckpointStore()
    run = _train_mcd(
        trained_backbone, store_factory, tracker, store_factory(persistent)
    )
    return run, tracker, persistent


@pytest.fixture
def probe_inputs(monkeypatch) -> list[np.ndarray]:
    """Record the inputs of every draws-per-second probe of a method run."""
    seen: list[np.ndarray] = []
    probe = method_run.draws_per_second

    def spy(
        draw: Callable[[jax.Array, jax.Array], jax.Array],
        inputs: jax.Array,
        key: jax.Array,
        num_draws: int,
    ) -> float:
        """Record the probed inputs, then time them with the real probe.

        Args:
            draw: The method run's draw function.
            inputs: The windows the probe times.
            key: The probe's draw key.
            num_draws: Draws per timed call.

        Returns:
            The real probe's draws per second.
        """
        seen.append(np.asarray(inputs))
        return probe(draw, inputs, key, num_draws)

    monkeypatch.setattr(method_run, "draws_per_second", spy)
    return seen


@pytest.mark.parametrize("config", [HNNConfig(), TINY_MCD], ids=["hnn", "mcd"])
def test_a_method_run_reads_only_the_posterior_validation_set(
    config,
    mcd_run,
    trained_backbone,
    only_serving,
    tracker_factory,
    store_factory,
    probe_inputs,
) -> None:
    # The posterior stage, its baseline and the draws-per-second probe all run on
    # the posterior validation set; reading the backbone one would fail.
    tracker = tracker_factory()
    data = trained_backbone.battery.data
    windows = np.asarray(data.posterior_val.features)

    train_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        data=only_serving(data, ValidationSet.POSTERIOR),
        config=config,
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store_factory(),
    )

    assert COST_DRAWS_PER_SECOND in tracker.tags
    # The probe times the first min(1024, n) posterior validation windows.
    assert DRAWS_PER_SECOND_WINDOWS == 1024
    (inputs,) = probe_inputs
    np.testing.assert_array_equal(inputs, windows[: min(1024, len(windows))])
    if isinstance(config, MCDConfig):
        _, reference, _ = mcd_run
        for tag in (CRPS, VAL_CRPS_BASELINE):
            assert tracker.history(tag) == pytest.approx(reference.history(tag))


@pytest.mark.parametrize("cap", [1, 3])
def test_the_draws_per_second_probe_times_only_the_first_windows(
    cap, trained_backbone, tracker_factory, store_factory, probe_inputs, monkeypatch
) -> None:
    # A posterior validation set larger than the window count is cut to its head.
    monkeypatch.setattr(method_run, "DRAWS_PER_SECOND_WINDOWS", cap)
    windows = np.asarray(trained_backbone.battery.data.posterior_val.features)
    assert len(windows) > cap

    train_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        data=trained_backbone.battery.data,
        config=HNNConfig(),
        seed=trained_backbone.seed,
        tracker=tracker_factory(),
        store=store_factory(),
    )

    (inputs,) = probe_inputs
    np.testing.assert_array_equal(inputs, windows[:cap])


def test_a_restored_mcd_run_yields_the_trained_runs_predictive_samples(
    mcd_run, trained_backbone, store_factory
) -> None:
    run, _, persistent = mcd_run
    restored = restore_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        config=TINY_MCD,
        seed=trained_backbone.seed,
        store=store_factory(persistent),
    )
    inputs = trained_backbone.battery.data.posterior_val.features[:7]
    key = jax.random.key(5)

    expected = run.predictive(inputs, key, num_draws=PREDICTIVE_SAMPLE_COUNT)
    actual = restored.predictive(inputs, key, num_draws=PREDICTIVE_SAMPLE_COUNT)

    assert actual.draws.shape == (PREDICTIVE_SAMPLE_COUNT, 7, 1)
    np.testing.assert_array_equal(actual.draws, expected.draws)


def test_the_same_seed_gives_identical_finalized_mcd_checkpoints_and_draws(
    mcd_run, trained_backbone, tracker_factory, store_factory, exactly_equal
) -> None:
    run, _, persistent = mcd_run
    again_persistent = InMemoryCheckpointStore()
    again = _train_mcd(
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


@settings(max_examples=10, deadline=None)
@given(start=st.integers(0, 20), size=st.integers(2, 6), split=st.integers(1, 5))
def test_mcd_draws_differ_but_each_is_one_function_across_input_subsets(
    mcd_run, trained_backbone, start: int, size: int, split: int
) -> None:
    run, _, _ = mcd_run
    split = min(split, size - 1)
    inputs = trained_backbone.battery.data.posterior_val.features[start : start + size]
    key = jax.random.key(3)

    whole = run.posterior.sample_means(inputs, key, 4)
    parts = np.concatenate(
        [
            run.posterior.sample_means(inputs[:split], key, 4),
            run.posterior.sample_means(inputs[split:], key, 4),
        ],
        axis=1,
    )
    repeated = run.posterior.sample_means(np.repeat(inputs[:1], size, axis=0), key, 4)

    np.testing.assert_allclose(parts, whole, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(repeated, np.repeat(repeated[:, :1], size, axis=1))
    for i in range(4):
        for j in range(i):
            assert not np.allclose(whole[i], whole[j])


def test_an_mcd_run_validates_on_crps_every_epoch_and_diagnoses_its_best_once(
    mcd_run, crps_per_epoch_and_diagnostics_once
) -> None:
    _, tracker, persistent = mcd_run

    crps_per_epoch_and_diagnostics_once(
        tracker, persistent.load("posterior/best").epoch
    )


def test_an_mcd_run_finalizes_its_best_epoch_on_validation_crps(mcd_run) -> None:
    run, tracker, persistent = mcd_run
    crps = tracker.history(CRPS)
    finalized = persistent.load("posterior/best")

    assert is_finalized(finalized)
    assert finalized.epoch == int(np.argmin(crps))
    assert run.checkpoint.epoch == finalized.epoch


def test_an_mcd_run_logs_its_cost_and_only_its_posterior_hyperparameters(
    mcd_run, trained_backbone
) -> None:
    _, tracker, _ = mcd_run
    network = trained_backbone.mean_hp.network
    conv_parameters = network.kernel_size * 2 * network.n_filters + network.n_filters

    assert COST_TAGS <= tracker.tags
    assert tracker.history(COST_TRAINABLE_PARAMETERS) == [float(conv_parameters)]
    assert set(tracker.params) == {"posterior"}
    assert tracker.params["posterior"] == {
        **TINY_MCD.posterior_hyperparameters(),
        "dataset_size": len(trained_backbone.battery.data.train),
    }


@settings(max_examples=10, deadline=None)
@given(start=st.integers(0, 20), size=st.integers(1, 6), seed=st.integers(0, 2**16))
def test_the_mcd_frozen_remainder_keeps_its_trained_mean_values(
    mcd_run, trained_backbone, start: int, size: int, seed: int
) -> None:
    run, _, _ = mcd_run
    mean_net = _merged(trained_backbone.checkpoints.mean.parameters)
    trained_conv = nnx.state(mean_net, nnx.All(nnx.Param, INFERRED_SUBNETWORK))
    inputs = trained_backbone.battery.data.posterior_val.features[start : start + size]
    key = jax.random.key(seed)
    # Dropout changes the output, so the reference is the trained mean network
    # under the same draw's mask: the MCD net with every Param at the trained
    # mean's values and its one dropout stream keyed as stagereg keys a draw
    # (stream i <- fold_in(key, i)).
    reference = build_mcd_net(
        trained_backbone.mean_hp.network, TINY_MCD.dropout_rate, jax.random.key(0)
    )
    nnx.update(reference, nnx.state(mean_net, nnx.Param))
    nnx.update(
        reference,
        jax.tree.map(
            lambda stream: stream.replace(jax.random.fold_in(key, 0)),
            nnx.state(reference, nnx.RngKey),
            is_leaf=lambda node: isinstance(node, nnx.Variable),
        ),
    )

    np.testing.assert_allclose(
        run.posterior.mean_function(trained_conv, inputs, key),
        reference(inputs),
        rtol=1e-6,
    )


def test_the_mcd_posterior_loss_decreases(mcd_run) -> None:
    _, tracker, _ = mcd_run
    losses = tracker.history("posterior/train/loss")

    assert losses[-1] < losses[0]


# --- Command-line overrides of a configuration -----------------------------------


def test_overrides_nest_the_schedule_and_named_posterior_params() -> None:
    overrides = posterior_overrides(
        ["burn_in_epochs=2", "avi/generator/learning_rate=0.01"],
        max_epochs=4,
        patience=None,
    )

    assert overrides == {
        "stage": {"max_epochs": 4},
        "burn_in_epochs": "2",
        "avi": {"generator": {"learning_rate": "0.01"}},
    }
    assert posterior_overrides([]) == {}


@pytest.mark.parametrize("assignment", ["burn_in_epochs", "=2"])
def test_a_malformed_override_is_refused(assignment: str) -> None:
    with pytest.raises(ValueError):
        posterior_overrides([assignment])


# --- the posterior validation baseline (val_crps_baseline) -------------------------


def test_a_bbb_run_logs_one_validation_baseline_before_its_first_epoch(
    bbb_run, one_baseline_before_epoch_0
) -> None:
    _, tracker, _ = bbb_run

    one_baseline_before_epoch_0(tracker)


def test_an_mcd_run_logs_one_validation_baseline_before_its_first_epoch(
    mcd_run, one_baseline_before_epoch_0
) -> None:
    _, tracker, _ = mcd_run

    one_baseline_before_epoch_0(tracker)


def test_an_hnn_run_logs_no_validation_baseline(hnn_run) -> None:
    _, tracker, _ = hnn_run

    assert VAL_CRPS_BASELINE not in tracker.tags


def _train_mcd_at(
    learning_rate: float, trained_backbone, tracker_factory, store_factory
):
    tracker = tracker_factory()
    persistent = InMemoryCheckpointStore()
    train_method_run(
        backbone=_fresh_backbone_handle(trained_backbone, store_factory),
        data=trained_backbone.battery.data,
        config=TINY_MCD.model_copy(
            update={"optimizer": OptimizerHyperparameters(learning_rate=learning_rate)}
        ),
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store_factory(persistent),
    )
    return tracker, persistent


def test_the_baseline_is_epoch_0s_score_of_a_posterior_that_barely_moved(
    trained_backbone, tracker_factory, store_factory
) -> None:
    tracker, _ = _train_mcd_at(1e-12, trained_backbone, tracker_factory, store_factory)

    assert tracker.history(VAL_CRPS_BASELINE)[0] == pytest.approx(
        tracker.history(CRPS)[0], rel=1e-5
    )


def test_the_best_epoch_ignores_a_baseline_lower_than_every_epoch(
    trained_backbone, tracker_factory, store_factory
) -> None:
    # An LR this large makes every epoch worse than the warm start.
    tracker, persistent = _train_mcd_at(
        10.0, trained_backbone, tracker_factory, store_factory
    )
    baseline = tracker.history(VAL_CRPS_BASELINE)[0]
    crps = tracker.history(CRPS)
    finalized = persistent.load("posterior/best")

    assert baseline < min(crps)
    assert finalized.epoch == int(np.argmin(crps))
    assert finalized.early_stopping_state.best_score == min(crps)


def test_the_posterior_wall_clock_excludes_the_validation_baseline(
    trained_backbone, tracker_factory, store_factory, monkeypatch
) -> None:
    # The baseline advances a fake clock by an hour; the cost must not see it.
    hour = 3600.0
    offset = 0.0
    real_perf_counter = method_run.time.perf_counter
    real_baseline = method_run._log_validation_baseline

    def slow_baseline(*args, **kwargs) -> None:
        nonlocal offset
        real_baseline(*args, **kwargs)
        offset += hour

    monkeypatch.setattr(
        method_run,
        "time",
        SimpleNamespace(perf_counter=lambda: real_perf_counter() + offset),
    )
    monkeypatch.setattr(method_run, "_log_validation_baseline", slow_baseline)
    tracker, _ = _train_mcd_at(1e-3, trained_backbone, tracker_factory, store_factory)

    assert 0 < tracker.history(COST_WALL_CLOCK)[0] < hour
