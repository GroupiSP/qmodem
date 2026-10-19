"""Experiment tracking on MLflow: the stagereg tracker, run setup and run context."""

from __future__ import annotations

import importlib.metadata
import logging
import os
import pathlib
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlflow
import pandas as pd
from matplotlib.figure import Figure
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from stagereg.core import flatten_parameters

from ..utils import ROOT_DIR


def stagereg_version() -> str:
    """Return the installed stagereg version, recorded as the ``stagereg_version`` tag.

    Returns:
        The version of the ``stagereg`` distribution.
    """
    return importlib.metadata.version("stagereg")


class MLflowTracker:
    """Stagereg ``ExperimentTracker`` that logs into one MLflow run.

    Metric tags are used verbatim as MLflow keys (``mean/train/loss``) with ``step``
    set to the stage epoch. Hyperparameters are flattened and grouped by stage
    (``mean/optimizer/learning_rate``).

    Attributes:
        run_id: The MLflow run logged into.
        client: The MLflow client used for logging.
    """

    def __init__(self, run_id: str, client: MlflowClient | None = None) -> None:
        """Bind the tracker to a run.

        Args:
            run_id: The MLflow run to log into. It need not be active.
            client: The MLflow client. Defaults to one on the current tracking URI.
        """
        self.run_id = run_id
        self.client = client if client is not None else MlflowClient()

    def log_params(self, values: Mapping[str, Any]) -> None:
        """Log nested hyperparameters, flattened by stage.

        Args:
            values: Hyperparameters nested by stage, e.g.
                ``{"mean": {"optimizer": {"learning_rate": 1e-3}}}``.
        """
        flat = flatten_parameters(values)
        self.client.log_batch(
            self.run_id,
            params=[Param(key, str(value)) for key, value in flat.items()],
        )

    def log_metrics(self, values: Mapping[str, float], *, step: int) -> None:
        """Log metric values under their stagereg tags.

        Args:
            values: Metric values keyed by stagereg tag (``stage/split/metric``).
            step: The stage epoch.
        """
        timestamp = int(time.time() * 1000)
        self.client.log_batch(
            self.run_id,
            metrics=[
                Metric(key, float(value), timestamp, step)
                for key, value in values.items()
            ],
        )

    def log_artifact(self, name: str, value: Any) -> None:
        """Log a text, a Matplotlib figure or a mapping as a run artifact.

        A figure without a file suffix is saved as ``<name>.png`` and a mapping as
        ``<name>.json``.

        Args:
            name: Artifact file path inside the run.
            value: A ``str`` (logged verbatim), a ``matplotlib.figure.Figure`` or a
                JSON-serializable ``Mapping``.

        Raises:
            TypeError: If ``value`` is of any other type.
        """
        if isinstance(value, str):
            self.client.log_text(self.run_id, value, name)
        elif isinstance(value, Figure):
            self.client.log_figure(self.run_id, value, _with_suffix(name, ".png"))
        elif isinstance(value, Mapping):
            self.client.log_dict(self.run_id, dict(value), _with_suffix(name, ".json"))
        else:
            raise TypeError(
                f"Cannot log artifact {name!r} of type {type(value).__name__}; "
                "expected str, matplotlib Figure or Mapping."
            )


def unflatten_parameters(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Nest flattened hyperparameters back by their ``/``-separated keys.

    It inverts stagereg's ``flatten_parameters``, e.g. for the params of a logged
    run: ``{"mean/optimizer/learning_rate": "0.001"}`` becomes
    ``{"mean": {"optimizer": {"learning_rate": "0.001"}}}``.

    Args:
        flat: Hyperparameters keyed by ``/``-separated paths.

    Returns:
        The nested hyperparameters.

    Raises:
        ValueError: If a key is both a leaf and a prefix of another key.
    """
    nested: dict[str, Any] = {}
    for key, value in flat.items():
        *path, leaf = key.split("/")
        node = nested
        for part in path:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Parameter {key!r} nests under a leaf.")
        if leaf in node:
            raise ValueError(f"Parameter {key!r} is both a leaf and a prefix.")
        node[leaf] = value
    return nested


def _with_suffix(name: str, suffix: str) -> str:
    """Append ``suffix`` to an artifact name that has no file suffix.

    Args:
        name: Artifact file path.
        suffix: Suffix to add, e.g. ``".json"``.

    Returns:
        ``name`` unchanged if it already has a suffix, else ``name + suffix``.
    """
    return name if pathlib.PurePosixPath(name).suffix else name + suffix


@dataclass(frozen=True)
class MLFlowSetup:
    """Configuration for an MLflow tracking run.

    Attributes:
        experiment_name: Name of the MLflow experiment to log under.
        run_name: Optional human-readable name for the MLflow run. If None, MLflow will auto-generate a name.
        run_id: Optional existing run ID to resume. If None, a new run is created.
        run_description: Optional description for the MLflow run.
        tags: Arbitrary key-value tags attached to the run.
        backend_store: SQLAlchemy URI for the MLflow backend store. Defaults to the value of the `MLFLOW_BACKEND_STORE` environment variable or `sqlite:///mlflow.db` if not set.
        artifact_store: Local path where artifacts are stored.
        tracking_server: Remote tracking server URI (not yet supported).
    """

    experiment_name: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    run_description: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    backend_store: str | None = None
    artifact_store: str | None = None
    tracking_server: str | None = None

    def __post_init__(self):
        if self.tracking_server is not None:
            raise NotImplementedError("Remote tracking server is not supported yet.")
        if self.experiment_name is None:
            object.__setattr__(
                self,
                "experiment_name",
                os.environ.get("MLFLOW_EXPERIMENT_NAME", "Default"),
            )
        if self.backend_store is None:
            object.__setattr__(
                self,
                "backend_store",
                os.environ.get("MLFLOW_BACKEND_STORE", "sqlite:///mlflow.db"),
            )
        if self.artifact_store is None:
            object.__setattr__(
                self,
                "artifact_store",
                os.environ.get("MLFLOW_ARTIFACT_STORE", ROOT_DIR / "mlruns"),
            )


@contextmanager
def track_mlflow(setup: MLFlowSetup) -> Generator[mlflow.ActiveRun, None, None]:
    """Run the enclosed block inside an MLflow run described by ``setup``.

    The tracking URI and experiment are set (the experiment is created with the
    setup's artifact store if needed), the run is started and tagged. The run ends as
    ``FINISHED``, or as ``FAILED`` if the block raises, so that tag-based run
    selection never picks an aborted run.

    Args:
        setup: Tracking backend, experiment, run name and tags.

    Yields:
        The active MLflow run.
    """
    mlflow.set_tracking_uri(setup.backend_store)

    exp_name = mlflow.get_experiment_by_name(setup.experiment_name)
    exp_id = (
        exp_name.experiment_id
        if exp_name is not None
        else mlflow.create_experiment(
            setup.experiment_name, artifact_location=str(setup.artifact_store)
        )
    )
    mlflow.set_experiment(experiment_id=exp_id)

    active_run = mlflow.start_run(
        run_id=setup.run_id,
        run_name=setup.run_name,
        description=setup.run_description,
    )
    try:
        mlflow.set_tags(setup.tags)
        yield active_run
    except BaseException:
        mlflow.end_run(status="FAILED")
        raise
    mlflow.end_run()


def track_dataframe(df: pd.DataFrame, name: str, context: str) -> None:
    """Log a dataframe as an input dataset of the active MLflow run.

    Args:
        df: The dataframe.
        name: The dataset's name, e.g. ``"battery_train"``.
        context: The dataset's role, e.g. ``"train"``.
    """
    dataset = mlflow.data.from_pandas(df, name=name)
    mlflow.log_input(dataset=dataset, context=context)


def log_window_counts(
    run_id: str,
    counts: Mapping[str, int],
    *,
    logger: logging.Logger,
    client: MlflowClient | None = None,
) -> None:
    """Report the window count of each split: log it and record it as run params.

    Args:
        run_id: The run receiving the counts as params.
        counts: Window count per split, keyed by param name (e.g.
            ``PreparedData.window_counts()``).
        logger: The logger the counts are reported to, at INFO level.
        client: The MLflow client. Defaults to one on the current tracking URI.
    """
    logger.info("Run %s, windows per split: %s", run_id, dict(counts))
    client = client if client is not None else MlflowClient()
    client.log_batch(
        run_id, params=[Param(key, str(value)) for key, value in counts.items()]
    )
