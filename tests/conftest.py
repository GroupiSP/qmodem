"""Shared session-scoped fixtures for the stagereg-based run pipeline.

The fixtures provide tiny synthetic battery windows, an in-memory tracker and checkpoint
store (stagereg protocol doubles), and a backbone trained once per session that later
slices can restore method runs from.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest
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
