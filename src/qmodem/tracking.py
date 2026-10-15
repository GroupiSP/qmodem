from __future__ import annotations

import argparse
import importlib.metadata
import io
import os
import pathlib
import pickle
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

import cloudpickle
import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
from matplotlib.figure import Figure
from mlflow.entities import Metric, Param
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient
from stagereg.core import (
    CachedCheckpointStore,
    Checkpoint,
    finalized_only,
    flatten_parameters,
)

from .data import DataScaler
from .utils import ROOT_DIR


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


def backbone_tags(*, seed: int, data_generation_run_id: str) -> dict[str, str]:
    """Return the tags of a backbone run.

    Args:
        seed: The protocol seed r.
        data_generation_run_id: MLflow run ID of the data-generation run.

    Returns:
        The ``role``, ``seed``, ``data_generation_run_id`` and ``stagereg_version``
        tags.
    """
    return {
        "role": "backbone",
        "seed": str(seed),
        "data_generation_run_id": data_generation_run_id,
        "stagereg_version": stagereg_version(),
    }


HPO_STUDY_ROLE = "hpo_study"
HPO_TRIAL_ROLE = "hpo_trial"
HPO_TRIAL_TAG = "hpo.trial"
PARENT_RUN_TAG = "mlflow.parentRunId"


def hpo_study_tags(
    *,
    stage: str,
    n_trials: int,
    method: str | None = None,
    backbone_run_id: str | None = None,
) -> dict[str, str]:
    """Return the tags of an HPO study run.

    Args:
        stage: The tuned stage, e.g. ``"mean"``, ``"variance"`` or ``"posterior"``.
        n_trials: The study's trial budget.
        method: The tuned method (posterior studies), or ``None`` for a backbone
            stage, in which case no ``method`` tag is set.
        backbone_run_id: The backbone run the trials restore, or ``None`` if they
            restore none (mean study), in which case no tag is set.

    Returns:
        The ``role``, ``stage``, ``n_trials``, ``stagereg_version`` and, where
        applicable, ``method`` and ``backbone_run_id`` tags.
    """
    optional = {"method": method, "backbone_run_id": backbone_run_id}
    return {
        "role": HPO_STUDY_ROLE,
        "stage": stage,
        "n_trials": str(n_trials),
        "stagereg_version": stagereg_version(),
        **{key: value for key, value in optional.items() if value is not None},
    }


def hpo_trial_tags(
    *, study_run_id: str, number: int, stage: str, method: str | None = None
) -> dict[str, str]:
    """Return the tags of an HPO trial run, a child of its study run.

    Args:
        study_run_id: The parent study run.
        number: The Optuna trial number.
        stage: The tuned stage.
        method: The tuned method, or ``None`` for a backbone stage.

    Returns:
        The ``role``, ``hpo.trial``, ``stage``, parent-run and, where applicable,
        ``method`` tags.
    """
    tags = {
        "role": HPO_TRIAL_ROLE,
        HPO_TRIAL_TAG: str(number),
        "stage": stage,
        PARENT_RUN_TAG: study_run_id,
    }
    return tags if method is None else {**tags, "method": method}


IDENTITY_TAGS: Mapping[str, tuple[str, ...]] = {
    "backbone": ("seed",),
    "method": ("method", "seed"),
}
"""Roles selectable by tags, and the tags that identify one run of each role.

HPO roles (``hpo_study``, ``hpo_trial``) are absent on purpose, so tag selection
never returns a tuning run.
"""


class RunSelectionError(LookupError):
    """Raised when run selection finds no run, or cannot tell which run is meant."""


@dataclass(frozen=True)
class RunCriteria:
    """Tag criteria selecting one protocol run.

    Attributes:
        role: The ``role`` tag, e.g. ``"backbone"`` or ``"method"``.
        seed: The ``seed`` tag (protocol seed r), or ``None`` to leave it open.
        method: The ``method`` tag, e.g. ``"q-avi"``, or ``None`` to leave it open.
    """

    role: str
    seed: int | None = None
    method: str | None = None


def resolve_run_id(
    *,
    run_id: str | None,
    criteria: RunCriteria | None,
    experiment_ids: Sequence[str],
    client: MlflowClient | None = None,
) -> str:
    """Return the run a script consumes: the explicit ``run_id`` or a tag selection.

    Args:
        run_id: An exact run ID (``--run-id``). It wins over ``criteria``.
        criteria: Tag criteria, used when ``run_id`` is ``None``.
        experiment_ids: Experiments searched by tag selection.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        The selected run's ID.

    Raises:
        ValueError: If neither ``run_id`` nor ``criteria`` is given, or the role is
            not selectable (see ``select_run``).
        RunSelectionError: If ``run_id`` does not exist, or tag selection finds no
            run or an ambiguous one (see ``select_run``).
    """
    client = client if client is not None else MlflowClient()
    if run_id is not None:
        try:
            return client.get_run(run_id).info.run_id
        except MlflowException as error:
            raise RunSelectionError(f"No run with ID {run_id!r}.") from error
    if criteria is None:
        raise ValueError("Pass a run ID or tag criteria to select a run.")
    return select_run(criteria, experiment_ids=experiment_ids, client=client)


@dataclass(frozen=True)
class RunSelector:
    """Command-line run selection for a script that consumes a run of one role.

    It adds an exact run-ID flag plus one flag per identity tag of the role
    (``--seed`` for a backbone, ``--method`` and ``--seed`` for a method run), and
    resolves the parsed arguments with ``resolve_run_id``.

    Attributes:
        role: The role of the consumed run, a key of ``IDENTITY_TAGS``.
        run_id_flag: The exact run-ID flag, e.g. ``"--backbone-run-id"``.
    """

    role: str
    run_id_flag: str = "--run-id"

    @property
    def _run_id_dest(self) -> str:
        return self.run_id_flag.lstrip("-").replace("-", "_")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the run-ID flag and the role's identity-tag flags.

        Args:
            parser: The script's parser.
        """
        parser.add_argument(
            self.run_id_flag,
            default=None,
            help=f"Exact {self.role} run ID; wins over tag selection.",
        )
        types = {"seed": int, "method": str}
        for tag in IDENTITY_TAGS[self.role]:
            parser.add_argument(
                f"--{tag}",
                type=types[tag],
                default=None,
                help=f"Select the newest finished {self.role} run by {tag}.",
            )

    def resolve(
        self,
        args: argparse.Namespace,
        *,
        experiment_ids: Sequence[str],
        client: MlflowClient | None = None,
    ) -> str:
        """Return the run selected by the parsed arguments.

        Args:
            args: Arguments parsed by a parser set up with ``add_arguments``.
            experiment_ids: Experiments searched by tag selection.
            client: The MLflow client. Defaults to one on the current tracking URI.

        Returns:
            The selected run's ID.

        Raises:
            RunSelectionError: See ``resolve_run_id``.
        """
        tags = {tag: getattr(args, tag) for tag in IDENTITY_TAGS[self.role]}
        return resolve_run_id(
            run_id=getattr(args, self._run_id_dest),
            criteria=RunCriteria(role=self.role, **tags),
            experiment_ids=experiment_ids,
            client=client,
        )


def _filter_string(criteria: RunCriteria) -> str:
    """Build the MLflow search filter of finished runs matching ``criteria``.

    Args:
        criteria: The tag criteria.

    Returns:
        The filter string.
    """
    tags = {"role": criteria.role, "seed": criteria.seed, "method": criteria.method}
    clauses = [
        f"tags.{key} = '{value}'" for key, value in tags.items() if value is not None
    ]
    return " AND ".join([*clauses, "attributes.status = 'FINISHED'"])


def select_run(
    criteria: RunCriteria,
    *,
    experiment_ids: Sequence[str],
    client: MlflowClient | None = None,
) -> str:
    """Select the newest finished run whose tags match ``criteria``.

    Args:
        criteria: The tag criteria.
        experiment_ids: Experiments to search.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Selection is ambiguous, and raises, when the criteria leave one of the role's
    identity tags (``IDENTITY_TAGS``) open, or when two or more finished matches
    share the newest start time. Older ties do not matter.

    Returns:
        The ID of the newest (latest start time) finished match.

    Raises:
        ValueError: If ``criteria.role`` is not selectable by tags (e.g.
            ``hpo_trial``); HPO runs are never selected as protocol runs.
        RunSelectionError: If no finished run matches, or the selection is
            ambiguous.
    """
    if criteria.role not in IDENTITY_TAGS:
        raise ValueError(
            f"Role {criteria.role!r} is not selectable by tags; "
            f"selectable roles are {sorted(IDENTITY_TAGS)}."
        )
    open_tags = [
        tag for tag in IDENTITY_TAGS[criteria.role] if getattr(criteria, tag) is None
    ]
    if open_tags:
        raise RunSelectionError(
            f"Ambiguous selection {criteria}: role {criteria.role!r} needs "
            f"{', '.join(open_tags)}."
        )
    client = client if client is not None else MlflowClient()
    runs = client.search_runs(
        list(experiment_ids),
        filter_string=_filter_string(criteria),
        order_by=["attributes.start_time DESC"],
        max_results=2,
    )
    if not runs:
        raise RunSelectionError(f"No finished run matches {criteria}.")
    if len(runs) == 2 and runs[0].info.start_time == runs[1].info.start_time:
        raise RunSelectionError(
            f"Ambiguous selection {criteria}: runs {runs[0].info.run_id} and "
            f"{runs[1].info.run_id} share the newest start time; pass --run-id."
        )
    return runs[0].info.run_id


@dataclass(frozen=True)
class ScalerArtifact:
    """A fitted data scaler to log as an MLflow pyfunc model.

    Attributes:
        name: Name of the logged model, e.g. ``"x_scaler"``.
        scaler: The fitted scaler.
        input_example: Example input, shaped like the data the scaler was fitted on.
        predict_method: Scaler method the pyfunc's ``predict`` calls, e.g.
            ``"inverse_transform"`` for the target scaler.
    """

    name: str
    scaler: DataScaler
    input_example: np.ndarray
    predict_method: str = "transform"


class _ScalerWrapper(mlflow.pyfunc.PythonModel):
    """Pyfunc model whose ``predict`` calls one method of a fitted scaler."""

    def __init__(self, scaler: DataScaler, predict_method: str = "transform") -> None:
        self.scaler = scaler
        self.predict_method = predict_method

    def predict(self, model_input: np.ndarray, params: Any = None) -> np.ndarray:
        return getattr(self.scaler, self.predict_method)(model_input)


class LoadedScaler:
    """Scikit-learn-like view of a scaler logged with ``log_scaler``.

    Both ``transform`` and ``inverse_transform`` call the pyfunc's ``predict``, whose
    behaviour was fixed at logging time by ``ScalerArtifact.predict_method``.

    Attributes:
        pyfunc: The loaded pyfunc model.
    """

    def __init__(self, pyfunc: PyFuncModel) -> None:
        """Wrap a loaded pyfunc model.

        Args:
            pyfunc: The pyfunc model logged by ``log_scaler``.
        """
        self.pyfunc = pyfunc

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the logged scaler method.

        Args:
            X: Input data.

        Returns:
            The scaled data.
        """
        return self.pyfunc.predict(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the logged scaler method.

        Args:
            X: Input data.

        Returns:
            The (inverse-)scaled data.
        """
        return self.pyfunc.predict(X)


def log_scaler(artifact: ScalerArtifact) -> None:
    """Log a fitted scaler as a pyfunc model of the active MLflow run.

    Args:
        artifact: The scaler and how to log it.
    """
    mlflow.pyfunc.log_model(
        name=artifact.name,
        python_model=_ScalerWrapper(
            artifact.scaler, predict_method=artifact.predict_method
        ),
        input_example=artifact.input_example,
    )


def load_scaler(run_id: str, name: str) -> LoadedScaler:
    """Load a scaler logged with ``log_scaler``.

    Method runs pass their parent backbone run's ID, where the scalers are logged
    once.

    Args:
        run_id: The run the scaler was logged into.
        name: The scaler's model name, e.g. ``"x_scaler"``.

    Returns:
        The loaded scaler.
    """
    return LoadedScaler(mlflow.pyfunc.load_model(f"runs:/{run_id}/{name}"))


def log_general(
    num_model_params: int,
    hyperparameters: dict[str, Any],
    scalers: Sequence[ScalerArtifact],
    log_stream: io.StringIO | None = None,
) -> None:
    """Log the parameter count, hyperparameters, scalers and log of the active run.

    Args:
        num_model_params: Number of trainable model parameters.
        hyperparameters: Flat hyperparameters.
        scalers: The fitted scalers.
        log_stream: The script's in-memory log, logged as ``training_log.txt``.
    """
    mlflow.log_param("n_params", num_model_params)
    mlflow.log_params(hyperparameters)
    for artifact in scalers:
        log_scaler(artifact)
    if log_stream is not None:
        mlflow.log_text(log_stream.getvalue(), "training_log.txt")


# TODO: implement
def get_tags_from_mlflow_run(run_id: str) -> dict[str, str]:
    pass


def track_dataframe(df: pd.DataFrame, name: str, context: str) -> None:
    dataset = mlflow.data.from_pandas(df, name=name)
    mlflow.log_input(dataset=dataset, context=context)


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
