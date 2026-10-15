"""Shared session-scoped fixtures for the stagereg-based run pipeline.

The fixtures provide tiny synthetic battery windows, an in-memory tracker and checkpoint
store (stagereg protocol doubles), and a backbone trained once per session that later
slices can restore method runs from.
"""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Iterator, Mapping
from typing import Any

import jax
import mlflow
import numpy as np
import pandas as pd
import pytest
from flax import nnx
from mlflow.tracking import MlflowClient
from stagereg.core import (
    CachedCheckpointStore,
    InMemoryCheckpointStore,
    TrainingEvent,
    finalized_only,
)

from qmodem.battery.data_processing import (
    WINDOW_SIZE,
    PreparedData,
    make_data_pipeline,
    prepare_windows,
)
from qmodem.battery.train import (
    BackboneCheckpoints,
    MeanHyperparameters,
    MeanNetworkHyperparameters,
    StageScheduleHyperparameters,
    VarianceHyperparameters,
    train_backbone,
)
from qmodem.tracking import MLFlowSetup

SYNTHETIC_TRAIN_LENGTHS: tuple[int, ...] = (23, 31, 27, 40)
SYNTHETIC_VAL_LENGTHS: tuple[int, ...] = (25, 19)


class InMemoryTracker:
    """In-memory ``ExperimentTracker`` and ``EventSink`` test double.

    stagereg 0.2.0 ships no in-memory tracker, so qmodem keeps its own copy of
    the one in stagereg's test suite.

    Attributes:
        events: Every training event received as an event sink.
        params: Logged hyperparameters (merged).
        metrics: Logged ``(metrics, step)`` pairs, in order.
        artifacts: Logged artifacts by name.
    """

    def __init__(self) -> None:
        """Create an empty tracker."""
        self.events: list[TrainingEvent] = []
        self.params: dict[str, Any] = {}
        self.metrics: list[tuple[dict[str, float], int]] = []
        self.artifacts: dict[str, Any] = {}

    def on_event(self, event: TrainingEvent) -> None:
        """Record a training event.

        Args:
            event: The event emitted by a stage.
        """
        self.events.append(event)

    def log_params(self, values: Mapping[str, Any]) -> None:
        """Record hyperparameters.

        Args:
            values: The (possibly nested) hyperparameters.
        """
        self.params.update(values)

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        """Record metric values at a step.

        Args:
            values: Metric values keyed by stagereg tag.
            step: The stage epoch.
        """
        self.metrics.append((dict(values), step))

    def log_artifact(self, name: str, value: Any) -> None:
        """Record an artifact.

        Args:
            name: Artifact name.
            value: Artifact payload.
        """
        self.artifacts[name] = value

    @property
    def tags(self) -> set[str]:
        """Every metric tag logged so far."""
        return {tag for values, _ in self.metrics for tag in values}

    def history(self, tag: str) -> list[float]:
        """Return the logged values of one metric tag, in step order.

        Args:
            tag: A stagereg metric tag, e.g. ``"mean/train/loss"``.

        Returns:
            The values logged under ``tag``.
        """
        return [values[tag] for values, _ in self.metrics if tag in values]


def _synthetic_histories(lengths: tuple[int, ...], seed: int) -> pd.DataFrame:
    """Build discharge histories with the columns of the battery CSV files.

    Args:
        lengths: Number of time steps of each history.
        seed: Seed of the feature noise.

    Returns:
        A dataframe with ``run_id``, ``time``, ``load``, ``voltage`` and ``rul``.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for run_id, length in enumerate(lengths):
        time = 5.0 * np.arange(length)
        load = 1.0 + 0.1 * run_id + 0.01 * rng.standard_normal(length)
        voltage = 4.2 - 0.02 * np.arange(length) * load
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "time": time,
                    "load": load,
                    "voltage": voltage,
                    "rul": time[-1] - time,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@dataclasses.dataclass(frozen=True)
class SyntheticBattery:
    """Tiny prepared battery data plus the facts it was built from.

    Attributes:
        data: Train and validation windows (window size ``WINDOW_SIZE``).
        train_lengths: Length of every training history.
        val_lengths: Length of every validation history.
    """

    data: PreparedData
    train_lengths: tuple[int, ...]
    val_lengths: tuple[int, ...]


@pytest.fixture(scope="session")
def synthetic_battery() -> SyntheticBattery:
    """Tiny synthetic battery windows prepared by the production pipeline."""
    data = prepare_windows(
        train_df=_synthetic_histories(SYNTHETIC_TRAIN_LENGTHS, seed=0),
        val_df=_synthetic_histories(SYNTHETIC_VAL_LENGTHS, seed=1),
        pipeline=make_data_pipeline(window_size=WINDOW_SIZE),
    )
    return SyntheticBattery(
        data=data,
        train_lengths=SYNTHETIC_TRAIN_LENGTHS,
        val_lengths=SYNTHETIC_VAL_LENGTHS,
    )


@pytest.fixture(scope="session")
def tracker_factory() -> type[InMemoryTracker]:
    """The in-memory tracker class (a factory, to keep Hypothesis happy)."""
    return InMemoryTracker


def make_store(
    persistent: InMemoryCheckpointStore | None = None,
) -> CachedCheckpointStore:
    """Build the in-memory checkpoint store wrapped as in production.

    Args:
        persistent: The store finalized checkpoints are forwarded to. A new
            in-memory store if ``None``.

    Returns:
        An in-memory store that persists only finalized checkpoints.
    """
    return CachedCheckpointStore(
        persistent if persistent is not None else InMemoryCheckpointStore(),
        persist_when=finalized_only,
    )


@pytest.fixture(scope="session")
def store_factory() -> Any:
    """A factory of in-memory checkpoint stores wrapped as in production."""
    return make_store


def assert_exactly_equal(actual: Any, expected: Any, path: str = "") -> None:
    """Assert that two checkpoint-like trees are equal, arrays compared exactly.

    Dataclasses, mappings, sequences and NNX variables are compared field by field.
    Arrays must agree in type, dtype, shape and every element. PRNG keys must agree
    in implementation and key data.

    Args:
        actual: The value under test.
        expected: The reference value.
        path: Location inside the tree, used in failure messages.

    Raises:
        AssertionError: If the trees differ anywhere.
    """
    assert type(actual) is type(expected), f"{path}: {type(actual)} != {type(expected)}"
    if isinstance(expected, jax.Array) and jax.dtypes.issubdtype(
        expected.dtype, jax.dtypes.prng_key
    ):
        assert jax.random.key_impl(actual) == jax.random.key_impl(expected), path
        assert_exactly_equal(
            jax.random.key_data(actual), jax.random.key_data(expected), path
        )
    elif isinstance(expected, (jax.Array, np.ndarray)):
        assert actual.dtype == expected.dtype, f"{path}: dtype"
        assert actual.shape == expected.shape, f"{path}: shape"
        assert np.array_equal(np.asarray(actual), np.asarray(expected)), path
    elif isinstance(expected, types.FunctionType):
        # Closures (e.g. optax's init/update functions) only compare by identity,
        # so compare their code and captured values instead.
        assert actual.__qualname__ == expected.__qualname__, path
        assert actual.__code__.co_code == expected.__code__.co_code, path
        assert_exactly_equal(
            [c.cell_contents for c in actual.__closure__ or ()],
            [c.cell_contents for c in expected.__closure__ or ()],
            f"{path}.__closure__",
        )
    elif isinstance(expected, nnx.Variable):
        assert_exactly_equal(actual.get_value(), expected.get_value(), path)
    elif dataclasses.is_dataclass(expected):
        for f in dataclasses.fields(expected):
            assert_exactly_equal(
                getattr(actual, f.name), getattr(expected, f.name), f"{path}.{f.name}"
            )
    elif isinstance(expected, Mapping):
        assert set(actual) == set(expected), f"{path}: keys"
        for key in expected:
            assert_exactly_equal(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"{path}: length"
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_exactly_equal(a, e, f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


@pytest.fixture(scope="session")
def exactly_equal() -> Any:
    """The exact tree comparison ``assert_exactly_equal``."""
    return assert_exactly_equal


@dataclasses.dataclass(frozen=True)
class MLflowBackend:
    """A file-backed MLflow tracking backend under a temporary directory.

    Attributes:
        setup: Points at the temporary backend store, artifact store and a
            test experiment.
        experiment_id: ID of the test experiment.
    """

    setup: MLFlowSetup
    experiment_id: str

    @property
    def client(self) -> MlflowClient:
        """A client bound to the temporary backend store."""
        return MlflowClient(tracking_uri=self.setup.backend_store)

    def new_run(self) -> str:
        """Create a run in the test experiment.

        Returns:
            The new run's ID.
        """
        return self.client.create_run(self.experiment_id).info.run_id


@pytest.fixture(scope="session")
def mlflow_backend(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[MLflowBackend]:
    """A file-backed MLflow backend, isolated from the repository's ``mlflow.db``.

    The global tracking URI points at the backend for the session and is restored
    afterwards.
    """
    root = tmp_path_factory.mktemp("mlflow")
    setup = MLFlowSetup(
        experiment_name="qmodem-tests",
        backend_store=f"sqlite:///{root / 'mlflow.db'}",
        artifact_store=str(root / "artifacts"),
    )
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(setup.backend_store)
    experiment_id = mlflow.create_experiment(
        setup.experiment_name, artifact_location=str(setup.artifact_store)
    )
    yield MLflowBackend(setup=setup, experiment_id=experiment_id)
    mlflow.set_tracking_uri(previous_uri)


TINY_MEAN_HP = MeanHyperparameters(
    network=MeanNetworkHyperparameters(kernel_size=3, n_filters=4, activation="gelu"),
    schedule=StageScheduleHyperparameters(max_epochs=6, patience=2),
    batch_size=16,
)
TINY_VARIANCE_HP = VarianceHyperparameters(
    schedule=StageScheduleHyperparameters(max_epochs=6, patience=2),
)
BACKBONE_SEED = 0


@dataclasses.dataclass(frozen=True)
class TrainedBackbone:
    """A backbone trained once on the synthetic windows.

    Attributes:
        battery: The synthetic data it was trained on.
        mean_hp: Mean-stage hyperparameters.
        variance_hp: Variance-stage hyperparameters.
        seed: The protocol seed r.
        tracker: The tracker the run logged into.
        store: The checkpoint store holding ``mean/best`` and ``variance/best``.
        persistent: The store behind ``store`` (only finalized checkpoints).
        checkpoints: The finalized checkpoints returned by ``train_backbone``.
    """

    battery: SyntheticBattery
    mean_hp: MeanHyperparameters
    variance_hp: VarianceHyperparameters
    seed: int
    tracker: InMemoryTracker
    store: CachedCheckpointStore
    persistent: InMemoryCheckpointStore
    checkpoints: BackboneCheckpoints


@pytest.fixture(scope="session")
def trained_backbone(synthetic_battery: SyntheticBattery) -> TrainedBackbone:
    """A tiny backbone trained once per session (seed ``BACKBONE_SEED``)."""
    tracker = InMemoryTracker()
    persistent = InMemoryCheckpointStore()
    store = make_store(persistent)
    checkpoints = train_backbone(
        data=synthetic_battery.data,
        mean_hp=TINY_MEAN_HP,
        variance_hp=TINY_VARIANCE_HP,
        seed=BACKBONE_SEED,
        tracker=tracker,
        store=store,
    )
    return TrainedBackbone(
        battery=synthetic_battery,
        mean_hp=TINY_MEAN_HP,
        variance_hp=TINY_VARIANCE_HP,
        seed=BACKBONE_SEED,
        tracker=tracker,
        store=store,
        persistent=persistent,
        checkpoints=checkpoints,
    )
