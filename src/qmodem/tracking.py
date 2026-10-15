from __future__ import annotations

import importlib.metadata
import io
import json
import os
import pathlib
import pickle
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Generator

import cloudpickle
import jax
import jax.numpy as jnp
import mlflow
import orbax.checkpoint as ocp
import pandas as pd
from matplotlib.figure import Figure
from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from stagereg.core import (
    CachedCheckpointStore,
    Checkpoint,
    finalized_only,
    flatten_parameters,
)

from .utils import LAST_TRAIN_SETUP_PATH, ROOT_DIR


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


CHECKPOINT_ARTIFACT_ROOT = "checkpoints"
_PICKLE_FILE = "checkpoint.pkl"
_ARRAYS_DIR = "arrays"


class _ArrayExtractingPickler(cloudpickle.CloudPickler):
    """Cloudpickler that moves every JAX array out of the pickle stream.

    Non-empty arrays are replaced by persistent IDs and collected in ``arrays``, so
    that they can be written with Orbax. Typed PRNG keys are stored as their key data
    plus the name of their implementation.
    """

    def __init__(self, file: io.BufferedIOBase, arrays: list[jax.Array]) -> None:
        super().__init__(file)
        self.arrays = arrays

    def persistent_id(self, obj: Any) -> tuple[str, int, str] | None:
        # Orbax refuses zero-size arrays; those stay in the pickle stream.
        if not isinstance(obj, jax.Array) or obj.size == 0:
            return None
        if jax.dtypes.issubdtype(obj.dtype, jax.dtypes.prng_key):
            self.arrays.append(jax.random.key_data(obj))
            return ("key", len(self.arrays) - 1, str(jax.random.key_impl(obj)))
        self.arrays.append(obj)
        return ("array", len(self.arrays) - 1, "")


class _ArrayRestoringUnpickler(pickle.Unpickler):
    """Unpickler that puts back the arrays extracted by ``_ArrayExtractingPickler``."""

    def __init__(
        self, file: io.BufferedIOBase, arrays: Mapping[str, jax.Array]
    ) -> None:
        super().__init__(file)
        self.arrays = arrays

    def persistent_load(self, pid: tuple[str, int, str]) -> jax.Array:
        kind, index, impl = pid
        array = jnp.asarray(self.arrays[str(index)])
        return jax.random.wrap_key_data(array, impl=impl) if kind == "key" else array


class MLflowCheckpointStore:
    """Stagereg ``CheckpointStore`` that keeps checkpoints as MLflow run artifacts.

    A checkpoint saved under ``key`` lives at ``checkpoints/<key>``: its arrays
    (including typed PRNG keys) are written with Orbax, the rest is cloudpickled
    (optimizer graph definitions hold Optax closures, which plain pickle refuses).
    Pickling is acceptable because stagereg is pinned and there is no backward
    compatibility. In production, wrap it with ``mlflow_checkpoint_store`` so that
    only finalized checkpoints are uploaded.

    Attributes:
        run_id: The MLflow run the checkpoints belong to.
        client: The MLflow client used for the artifacts.
    """

    def __init__(self, run_id: str, client: MlflowClient | None = None) -> None:
        """Bind the store to a run.

        Args:
            run_id: The MLflow run holding the checkpoints. It need not be active.
            client: The MLflow client. Defaults to one on the current tracking URI.
        """
        self.run_id = run_id
        self.client = client if client is not None else MlflowClient()

    def save(self, key: str, checkpoint: Checkpoint) -> None:
        """Upload a checkpoint to ``checkpoints/<key>``.

        Args:
            key: Checkpoint key, e.g. ``"mean/best"``.
            checkpoint: The checkpoint to save.
        """
        arrays: list[jax.Array] = []
        stream = io.BytesIO()
        _ArrayExtractingPickler(stream, arrays).dump(checkpoint)
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(tmp)
            (directory / _PICKLE_FILE).write_bytes(stream.getvalue())
            if arrays:
                checkpointer = ocp.StandardCheckpointer()
                checkpointer.save(
                    directory / _ARRAYS_DIR,
                    {str(i): array for i, array in enumerate(arrays)},
                )
                checkpointer.wait_until_finished()
            self.client.log_artifacts(
                self.run_id, str(directory), artifact_path=self._artifact_path(key)
            )

    def load(self, key: str) -> Checkpoint:
        """Download and rebuild the checkpoint saved under ``key``.

        Args:
            key: Checkpoint key, e.g. ``"mean/best"``.

        Returns:
            The saved checkpoint, with arrays restored exactly.

        Raises:
            KeyError: If no checkpoint is saved under ``key``.
        """
        if not self.exists(key):
            raise KeyError(key)
        with tempfile.TemporaryDirectory() as tmp:
            directory = pathlib.Path(
                self.client.download_artifacts(
                    self.run_id, self._artifact_path(key), dst_path=tmp
                )
            )
            arrays: Mapping[str, jax.Array] = {}
            if (directory / _ARRAYS_DIR).exists():
                arrays = ocp.StandardCheckpointer().restore(
                    (directory / _ARRAYS_DIR).resolve()
                )
            with (directory / _PICKLE_FILE).open("rb") as file:
                return _ArrayRestoringUnpickler(file, arrays).load()

    def exists(self, key: str) -> bool:
        """Return whether a checkpoint is saved under ``key``.

        Args:
            key: Checkpoint key, e.g. ``"mean/best"``.

        Returns:
            ``True`` if ``checkpoints/<key>`` holds a checkpoint.
        """
        artifacts = self.client.list_artifacts(self.run_id, self._artifact_path(key))
        return any(a.path.endswith(_PICKLE_FILE) for a in artifacts)

    @staticmethod
    def _artifact_path(key: str) -> str:
        return f"{CHECKPOINT_ARTIFACT_ROOT}/{key}"


def mlflow_checkpoint_store(
    run_id: str, client: MlflowClient | None = None
) -> CachedCheckpointStore:
    """Build the production checkpoint store of a run.

    Every save is cached in memory, and only finalized checkpoints are uploaded to
    the run's MLflow artifacts.

    Args:
        run_id: The MLflow run holding the checkpoints.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        ``MLflowCheckpointStore`` wrapped in ``CachedCheckpointStore`` with
        ``persist_when=finalized_only``.
    """
    return CachedCheckpointStore(
        MLflowCheckpointStore(run_id, client=client), persist_when=finalized_only
    )


def _with_suffix(name: str, suffix: str) -> str:
    """Append ``suffix`` to an artifact name that has no file suffix.

    Args:
        name: Artifact file path.
        suffix: Suffix to add, e.g. ``".json"``.

    Returns:
        ``name`` unchanged if it already has a suffix, else ``name + suffix``.
    """
    return name if pathlib.PurePosixPath(name).suffix else name + suffix


class DatasetChoice(StrEnum):
    BATTERY = auto()


class LossChoice(StrEnum):
    MSE = auto()
    NLL = auto()


class ModelChoice(StrEnum):
    CNN = auto()


class OptimizerChoice(StrEnum):
    ADAM = auto()
    SGD = auto()


class SchedulerChoice(StrEnum):
    COSINE = auto()
    STEP = auto()


class HPSamplerChoice(StrEnum):
    RANDOM = auto()
    TPE = auto()


class HPPrunerChoice(StrEnum):
    ASHA = auto()
    MEDIAN = auto()
    NOP = auto()


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

    try:
        active_run = mlflow.start_run(
            run_id=setup.run_id,
            run_name=setup.run_name,
            description=setup.run_description,
        )
        mlflow.set_tags(setup.tags)

        yield active_run

    finally:
        mlflow.end_run()


# TODO: implement
def get_tags_from_mlflow_run(run_id: str) -> dict[str, str]:
    pass


def track_dataframe(df: pd.DataFrame, name: str, context: str) -> None:
    dataset = mlflow.data.from_pandas(df, name=name)
    mlflow.log_input(dataset=dataset, context=context)


def write_setup_to_file() -> None:
    """Writes only the setup information necessary to resume a trained run for testing
    purposes."""
    active_run = mlflow.active_run()
    active_experiment = mlflow.get_experiment(active_run.info.experiment_id)

    with open(LAST_TRAIN_SETUP_PATH, "w") as f:
        json.dump(
            {
                "run_id": active_run.info.run_id,
                "experiment_name": active_experiment.name,
            },
            f,
        )

    return


def retrieve_mlflow_setup_train() -> MLFlowSetup:
    """Retrieves the MLFlowSetup from a JSON file at the given path. If the file does
    not exist, prompts the user for input.

    Returns:
        An instance of MLFlowSetup.
    """
    use_last = os.environ.get("MLFLOW_USE_LAST_TRAINED", "").lower() in ("1", "true")
    if use_last:
        with open(LAST_TRAIN_SETUP_PATH, "r") as f:
            data = json.load(f)
        return MLFlowSetup(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
        )
    else:
        print(
            "MLFLOW_USE_LAST_TRAINED is set to False. Please input the training run ID and experiment name."
        )
        run_id = input("Enter the training run ID: ").strip()
        experiment_name = input("Enter the experiment name: ").strip()
        return MLFlowSetup(
            run_id=run_id,
            experiment_name=experiment_name,
        )


def get_run_parameters(run_id: str, backend_store: str) -> dict[str, Any]:
    """Retrieves the parameters of a specific MLflow run.

    Args:
        run_id (str): The ID of the MLflow run.

    Returns:
        dict[str, Any]: A dictionary containing the parameters of the run.
    """
    mlflow.set_tracking_uri(backend_store)
    run = mlflow.get_run(run_id)
    return run.data.params
