from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import sklearn.preprocessing as skpp
from flax import nnx
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from stagereg.core import Checkpoint, flatten_parameters, is_finalized

from qmodem.battery.data_processing import FEATURES, WINDOW_SIZE, ValidationSet
from qmodem.battery.methods.mcd import MCDConfig
from qmodem.battery.train import (
    MeanHyperparameters,
    MeanNetworkHyperparameters,
    OptimizerHyperparameters,
    StageScheduleHyperparameters,
    TrackedBackbone,
    backbone_keys,
    backbone_optimizer,
    load_backbone_hyperparameters,
    steps_per_epoch,
    track_backbone_run,
    train_backbone,
    train_mean_stage,
    train_variance_stage,
)
from qmodem.tracking import (
    MLflowCheckpointStore,
    ScalerArtifact,
    Stage,
    unflatten_parameters,
)

# --- Backbone run ---------------------------------------------------------


def _leaves(checkpoint: Checkpoint) -> list[np.ndarray]:
    """Every array a checkpoint carries: weights, optimizer state and RNG key."""
    trees = (
        checkpoint.parameters.state,
        checkpoint.optimizer_state.state,
        jax.random.key_data(checkpoint.state.rng_state),
    )
    return [np.asarray(leaf) for leaf in jax.tree.leaves(trees)]


def _assert_bit_identical(first: Checkpoint, second: Checkpoint) -> None:
    first_leaves, second_leaves = _leaves(first), _leaves(second)
    assert len(first_leaves) == len(second_leaves)
    for a, b in zip(first_leaves, second_leaves, strict=True):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)


def test_mean_hyperparameters_default_to_the_pre_study_backbone() -> None:
    network = MeanHyperparameters().network

    assert (network.kernel_size, network.n_filters, network.activation) == (
        3,
        29,
        "gelu",
    )


def test_mean_hyperparameters_are_frozen() -> None:
    hp = MeanHyperparameters()

    with pytest.raises(ValidationError):
        hp.batch_size = 64


def test_mean_network_rejects_unknown_activation() -> None:
    with pytest.raises(ValidationError):
        MeanNetworkHyperparameters(activation="softmax")


_mean_hyperparameters = st.builds(
    MeanHyperparameters,
    network=st.builds(
        MeanNetworkHyperparameters,
        kernel_size=st.integers(2, 8),
        n_filters=st.integers(4, 64),
        activation=st.sampled_from(["gelu", "relu", "silu", "tanh"]),
    ),
    optimizer=st.builds(OptimizerHyperparameters, learning_rate=st.floats(1e-5, 1e-1)),
    schedule=st.builds(
        StageScheduleHyperparameters,
        max_epochs=st.integers(1, 1000),
        patience=st.integers(0, 100),
    ),
    batch_size=st.integers(1, 256),
)


@settings(deadline=None)
@given(hp=_mean_hyperparameters)
def test_mean_hyperparameters_round_trip_through_their_logged_params(
    hp: MeanHyperparameters,
) -> None:
    flat = flatten_parameters({Stage.MEAN: hp.model_dump()})
    logged = {name: str(value) for name, value in flat.items()}

    nested = unflatten_parameters(logged)

    assert MeanHyperparameters.model_validate(nested[Stage.MEAN]) == hp


@settings(deadline=None)
@given(seed=st.integers(0, 2**31 - 1), other=st.integers(0, 2**31 - 1))
def test_backbone_keys_are_reproducible_and_distinct_across_streams_and_seeds(
    seed: int, other: int
) -> None:
    keys = dataclasses.astuple(backbone_keys(seed))
    again = dataclasses.astuple(backbone_keys(seed))
    data = [tuple(np.asarray(jax.random.key_data(k)).ravel()) for k in keys]

    assert all(
        np.array_equal(jax.random.key_data(a), jax.random.key_data(b))
        for a, b in zip(keys, again, strict=True)
    )
    assert len(set(data)) == len(data)
    if other != seed:
        others = dataclasses.astuple(backbone_keys(other))
        assert not np.array_equal(
            jax.random.key_data(others[0]), jax.random.key_data(keys[0])
        )


def test_a_backbone_run_is_selected_on_the_backbone_validation_set(
    trained_backbone, only_serving, tracker_factory, store_factory
) -> None:
    data = only_serving(trained_backbone.battery.data, ValidationSet.BACKBONE)
    tracker = tracker_factory()

    train_backbone(
        data=data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store_factory(),
    )

    for stage in ("mean", "variance"):
        assert tracker.history(f"{stage}/validation/loss") == pytest.approx(
            trained_backbone.tracker.history(f"{stage}/validation/loss")
        )


# --- Mean and variance stage optimizer: per-epoch cosine LR decay ---------


def _step_sizes(
    transformation: optax.GradientTransformation, n_steps: int
) -> list[float]:
    """Adam's step size at each of ``n_steps`` steps under a constant unit gradient.

    Under a constant gradient Adam's bias-corrected update is ``lr / (1 + eps)``,
    so the step size reads off the learning rate of every step.
    """
    parameters = jnp.zeros(())
    gradient = jnp.ones(())
    opt_state = transformation.init(parameters)
    sizes = []
    for _ in range(n_steps):
        updates, opt_state = transformation.update(gradient, opt_state, parameters)
        sizes.append(float(-updates))
    return sizes


@settings(deadline=None, max_examples=25)
@given(
    learning_rate=st.floats(1e-5, 1e-1),
    max_epochs=st.integers(1, 12),
    steps_per_epoch=st.integers(1, 4),
)
def test_backbone_optimizer_decays_the_learning_rate_per_epoch_to_one_percent(
    learning_rate: float, max_epochs: int, steps_per_epoch: int
) -> None:
    transformation = backbone_optimizer(
        OptimizerHyperparameters(learning_rate=learning_rate),
        max_epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    sizes = _step_sizes(transformation, (max_epochs + 1) * steps_per_epoch)
    per_epoch = [
        sizes[e * steps_per_epoch : (e + 1) * steps_per_epoch]
        for e in range(max_epochs + 1)
    ]

    assert sizes[0] == pytest.approx(learning_rate, rel=1e-4)
    assert per_epoch[max_epochs][0] == pytest.approx(0.01 * learning_rate, rel=1e-4)
    assert all(b <= a * (1 + 1e-4) for a, b in zip(sizes, sizes[1:], strict=False))
    assert all(
        epoch == pytest.approx([epoch[0]] * len(epoch), rel=1e-4) for epoch in per_epoch
    )


@settings(deadline=None, max_examples=10)
@given(learning_rate=st.floats(1e-5, 1e-1))
def test_the_posterior_optimizer_keeps_a_constant_learning_rate(
    learning_rate: float,
) -> None:
    config = MCDConfig(optimizer=OptimizerHyperparameters(learning_rate=learning_rate))
    method = config.posterior_spec().build_method(jax.random.key(0))

    sizes = _step_sizes(method.optimizer, 20)

    assert sizes == pytest.approx([learning_rate] * 20, rel=1e-4)


@given(batch_size=st.integers(-8, 0))
def test_a_non_positive_batch_size_has_no_steps_per_epoch(
    synthetic_battery, batch_size: int
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        steps_per_epoch(synthetic_battery.data, batch_size)


@given(batch_size=st.integers(1, 200))
def test_every_training_window_fits_in_the_steps_of_an_epoch(
    synthetic_battery, batch_size: int
) -> None:
    n_windows = len(synthetic_battery.data.train)

    steps = steps_per_epoch(synthetic_battery.data, batch_size)

    assert steps * batch_size >= n_windows
    assert steps == 1 or (steps - 1) * batch_size < n_windows


def test_backbone_returns_finalized_mean_and_variance_checkpoints(
    trained_backbone,
) -> None:
    checkpoints = trained_backbone.checkpoints

    assert is_finalized(checkpoints.mean)
    assert is_finalized(checkpoints.variance)
    assert checkpoints.mean.metadata["stage"] == "mean"
    assert checkpoints.variance.metadata["stage"] == "variance"


def test_backbone_persists_only_its_finalized_checkpoints(trained_backbone) -> None:
    persistent = trained_backbone.persistent

    for key in ("mean/best", "variance/best"):
        assert is_finalized(persistent.load(key))
    assert not persistent.exists("posterior/best")


def test_backbone_logs_both_stages_losses(trained_backbone) -> None:
    assert {
        "mean/train/loss",
        "mean/validation/loss",
        "variance/train/loss",
        "variance/validation/loss",
    } <= trained_backbone.tracker.tags


def test_backbone_logs_hyperparameters_grouped_by_stage(trained_backbone) -> None:
    params = trained_backbone.tracker.params

    assert params["mean"] == trained_backbone.mean_hp.model_dump()
    assert params["variance"] == trained_backbone.variance_hp.model_dump()


def test_same_seed_gives_bit_identical_backbone_checkpoints(
    trained_backbone, tracker_factory, store_factory
) -> None:
    again = train_backbone(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        tracker=tracker_factory(),
        store=store_factory(),
    )

    _assert_bit_identical(again.mean, trained_backbone.checkpoints.mean)
    _assert_bit_identical(again.variance, trained_backbone.checkpoints.variance)


def test_another_seed_gives_another_backbone(
    trained_backbone, tracker_factory, store_factory
) -> None:
    other = train_backbone(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed + 1,
        tracker=tracker_factory(),
        store=store_factory(),
    )

    first_weights = jax.tree.leaves(other.mean.parameters.state)
    second_weights = jax.tree.leaves(trained_backbone.checkpoints.mean.parameters.state)
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(first_weights, second_weights, strict=True)
    )


@pytest.mark.parametrize("stage", ["mean", "variance"])
def test_backbone_training_loss_decreases(trained_backbone, stage: str) -> None:
    losses = trained_backbone.tracker.history(f"{stage}/train/loss")

    assert losses[-1] < losses[0]


# --- Single backbone stages (HPO trials train one stage) ---------------------------


def test_the_mean_stage_alone_reproduces_the_backbone_mean(
    trained_backbone, tracker_factory, store_factory
) -> None:
    tracker, store = tracker_factory(), store_factory()

    mean = train_mean_stage(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store,
    )

    _assert_bit_identical(mean, trained_backbone.checkpoints.mean)
    assert not store.exists("variance/best")
    assert tracker.params == {"mean": trained_backbone.mean_hp.model_dump()}
    assert {tag.split("/")[0] for tag in tracker.tags} == {"mean"}


def test_the_variance_stage_on_the_backbone_mean_reproduces_the_backbone_variance(
    trained_backbone, tracker_factory, store_factory
) -> None:
    tracker, store = tracker_factory(), store_factory()

    variance = train_variance_stage(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        mean_checkpoint=trained_backbone.checkpoints.mean,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        tracker=tracker,
        store=store,
    )

    _assert_bit_identical(variance, trained_backbone.checkpoints.variance)
    assert not store.exists("mean/best")
    assert tracker.params == {"variance": trained_backbone.variance_hp.model_dump()}
    assert {tag.split("/")[0] for tag in tracker.tags} == {"variance"}


def test_extra_event_sinks_see_every_validation_epoch(
    trained_backbone, tracker_factory, store_factory
) -> None:
    seen = tracker_factory()

    train_mean_stage(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        seed=trained_backbone.seed,
        tracker=tracker_factory(),
        store=store_factory(),
        event_sinks=(seen,),
    )

    validation = [e for e in seen.events if e.name == "validation_end"]
    assert [e.step for e in validation] == list(range(len(validation)))
    assert validation


# --- Backbone run tracked in MLflow (seam 2) --------------------------------------

_DATA_GENERATION_RUN_ID = "data-generation-run"


@dataclasses.dataclass(frozen=True)
class _MLflowBackbone:
    run: TrackedBackbone
    scalers: tuple[ScalerArtifact, ...]
    backend: Any
    reference: Any


@pytest.fixture(scope="module")
def mlflow_backbone(mlflow_backend, trained_backbone) -> _MLflowBackbone:
    """A tiny backbone run tracked in the file-backed MLflow backend."""
    rng = np.random.default_rng(0)
    x_scaler = skpp.StandardScaler().fit(rng.normal(3.0, 2.0, (50, len(FEATURES))))
    y_scaler = skpp.MinMaxScaler().fit(rng.uniform(0.0, 900.0, (50, 1)))
    scalers = (
        ScalerArtifact(
            name="x_scaler",
            scaler=x_scaler,
            input_example=np.ones((WINDOW_SIZE, len(FEATURES))),
        ),
        ScalerArtifact(
            name="y_scaler",
            scaler=y_scaler,
            input_example=np.ones((1, 1)),
            predict_method="inverse_transform",
        ),
    )
    run = track_backbone_run(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        data_generation_run_id=_DATA_GENERATION_RUN_ID,
        scalers=scalers,
        setup=mlflow_backend.setup,
    )
    return _MLflowBackbone(
        run=run, scalers=scalers, backend=mlflow_backend, reference=trained_backbone
    )


def test_the_backbone_run_carries_its_four_tags(mlflow_backbone) -> None:
    run = mlflow_backbone.backend.client.get_run(mlflow_backbone.run.run_id)

    assert {
        key: run.data.tags.get(key)
        for key in ("role", "seed", "data_generation_run_id", "stagereg_version")
    } == {
        "role": "backbone",
        "seed": str(mlflow_backbone.reference.seed),
        "data_generation_run_id": _DATA_GENERATION_RUN_ID,
        "stagereg_version": "0.4.0",
    }
    assert run.info.status == "FINISHED"


def test_only_the_finalized_backbone_checkpoints_reach_mlflow(
    mlflow_backbone, exactly_equal
) -> None:
    client = mlflow_backbone.backend.client
    run_id = mlflow_backbone.run.run_id
    store = MLflowCheckpointStore(run_id, client=client)

    stage_dirs = {a.path for a in client.list_artifacts(run_id, "checkpoints")}
    assert stage_dirs == {"checkpoints/mean", "checkpoints/variance"}
    for stage in ("mean", "variance"):
        assert {
            a.path for a in client.list_artifacts(run_id, f"checkpoints/{stage}")
        } == {f"checkpoints/{stage}/best"}
    exactly_equal(store.load("mean/best"), mlflow_backbone.run.checkpoints.mean)
    exactly_equal(store.load("variance/best"), mlflow_backbone.run.checkpoints.variance)
    assert is_finalized(store.load("mean/best"))
    assert is_finalized(store.load("variance/best"))


@pytest.mark.parametrize("stage", ["mean", "variance"])
def test_backbone_losses_arrive_with_the_stage_epoch_as_step(
    mlflow_backbone, stage: str
) -> None:
    history = mlflow_backbone.backend.client.get_metric_history(
        mlflow_backbone.run.run_id, f"{stage}/train/loss"
    )

    steps = sorted(metric.step for metric in history)
    assert steps == list(range(len(steps)))
    assert len(steps) >= 2


def test_backbone_hyperparameters_arrive_flattened_by_stage(mlflow_backbone) -> None:
    params = mlflow_backbone.backend.client.get_run(
        mlflow_backbone.run.run_id
    ).data.params
    mean_hp = mlflow_backbone.reference.mean_hp
    variance_hp = mlflow_backbone.reference.variance_hp

    assert params["mean/optimizer/learning_rate"] == str(
        mean_hp.optimizer.learning_rate
    )
    assert params["mean/network/n_filters"] == str(mean_hp.network.n_filters)
    assert params["variance/schedule/patience"] == str(variance_hp.schedule.patience)


def test_backbone_hyperparameters_load_back_from_the_run(mlflow_backbone) -> None:
    loaded = load_backbone_hyperparameters(
        mlflow_backbone.run.run_id, client=mlflow_backbone.backend.client
    )

    assert loaded == (
        mlflow_backbone.reference.mean_hp,
        mlflow_backbone.reference.variance_hp,
    )


def test_the_tracked_backbone_matches_the_untracked_one(
    mlflow_backbone, exactly_equal
) -> None:
    exactly_equal(
        mlflow_backbone.run.checkpoints.mean.parameters,
        mlflow_backbone.reference.checkpoints.mean.parameters,
    )


_FRESH_PROCESS_RESTORE = """
import sys

import jax.numpy as jnp
import numpy as np
from flax import nnx

from qmodem.tracking import MLflowCheckpointStore, load_scaler

run_id, out = sys.argv[1], sys.argv[2]
store = MLflowCheckpointStore(run_id)
inputs = jnp.asarray(np.load(out)["inputs"])
mean = store.load("mean/best").parameters
variance = store.load("variance/best").parameters
np.savez(
    out,
    mean=np.asarray(nnx.merge(mean.graphdef, mean.state)(inputs)),
    variance=np.asarray(nnx.merge(variance.graphdef, variance.state)(inputs).mean()),
    x_scaled=load_scaler(run_id, "x_scaler").transform(np.load(out)["features"]),
    y_rul=load_scaler(run_id, "y_scaler").inverse_transform(np.load(out)["targets"]),
)
"""


def test_backbone_checkpoints_and_scalers_restore_in_a_fresh_process(
    mlflow_backbone, tmp_path
) -> None:
    inputs = np.asarray(
        mlflow_backbone.reference.battery.data.posterior_val.features[:5]
    )
    features = np.linspace(0.5, 1.5, 2 * WINDOW_SIZE).reshape(WINDOW_SIZE, 2)
    targets = np.linspace(0.0, 1.0, 4).reshape(4, 1)
    out = tmp_path / "restored.npz"
    np.savez(out, inputs=inputs, features=features, targets=targets)

    subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_RESTORE, mlflow_backbone.run.run_id, out],
        check=True,
        env={
            **os.environ,
            "MLFLOW_TRACKING_URI": mlflow_backbone.backend.setup.backend_store,
        },
    )

    restored = np.load(out)
    checkpoints = mlflow_backbone.run.checkpoints
    mean = nnx.merge(
        checkpoints.mean.parameters.graphdef, checkpoints.mean.parameters.state
    )
    variance = nnx.merge(
        checkpoints.variance.parameters.graphdef, checkpoints.variance.parameters.state
    )
    np.testing.assert_array_equal(restored["mean"], np.asarray(mean(inputs)))
    np.testing.assert_array_equal(
        restored["variance"], np.asarray(variance(inputs).mean())
    )
    x_scaler, y_scaler = (s.scaler for s in mlflow_backbone.scalers)
    np.testing.assert_allclose(restored["x_scaled"], x_scaler.transform(features))
    np.testing.assert_allclose(restored["y_rul"], y_scaler.inverse_transform(targets))
