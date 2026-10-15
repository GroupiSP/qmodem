"""Stagereg checkpoints stored as MLflow run artifacts."""

from __future__ import annotations

import importlib
import io
import pathlib
import pickle
import sys
import tempfile
import types
from collections.abc import Mapping
from typing import Any

import cloudpickle
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from mlflow.tracking import MlflowClient
from stagereg.core import CachedCheckpointStore, Checkpoint, finalized_only


CHECKPOINT_ARTIFACT_ROOT = "checkpoints"
_PICKLE_FILE = "checkpoint.pkl"
_ARRAYS_DIR = "arrays"


def _import_attribute(module: str, qualname: str) -> Any:
    """Look up ``module.qualname`` (the unpickling side of ``_importable_reference``).

    Args:
        module: Name of the module holding the object.
        qualname: Dotted qualified name of the object inside the module.

    Returns:
        The object.
    """
    obj: Any = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _importable_reference(obj: Any) -> tuple[str, str] | None:
    """Return ``(module, qualname)`` if a callable object can be imported back.

    Plain functions and classes are left to cloudpickle. Other callables, such as
    JAX's jitted or ``custom_jvp`` activations (``nnx.relu``, ``nnx.tanh``), would be
    pickled by value and load as a different object, which breaks graph equality
    when a stage restores a checkpoint.

    Args:
        obj: Any object reached while pickling.

    Returns:
        The reference, or ``None`` if ``obj`` must be pickled normally.
    """
    if isinstance(obj, (type, types.FunctionType)) or not callable(obj):
        return None
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        return None
    if module not in sys.modules or "<locals>" in qualname:
        return None
    try:
        found = _import_attribute(module, qualname)
    except AttributeError:
        return None
    return (module, qualname) if found is obj else None


class _ArrayExtractingPickler(cloudpickle.CloudPickler):
    """Cloudpickler that moves every JAX array out of the pickle stream.

    Non-empty arrays are replaced by persistent IDs and collected in ``arrays``, so
    that they can be written with Orbax. Typed PRNG keys are stored as their key data
    plus the name of their implementation. Importable callables that are not plain
    functions (e.g. jitted activations) are pickled by reference, so that they load
    as the same object.
    """

    def __init__(self, file: io.BufferedIOBase, arrays: list[jax.Array]) -> None:
        super().__init__(file)
        self.arrays = arrays

    def reducer_override(self, obj: Any) -> Any:
        """Pickle importable non-function callables by reference.

        Args:
            obj: The object being pickled.

        Returns:
            A reduce tuple, or whatever cloudpickle would do otherwise.
        """
        reference = _importable_reference(obj)
        if reference is not None:
            return (_import_attribute, reference)
        return super().reducer_override(obj)

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
