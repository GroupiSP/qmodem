from __future__ import annotations

import dataclasses
import functools
import pathlib
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp
from stagereg.core import Batch, LoaderFactory

from qmodem.data import (
    ArrayDataSource,
    DataPipeline,
    DataScaler,
    ScalingMode,
    ScalingStep,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    to_jax,
)

WINDOW_SIZE = 10
"""Window size shared by every method, so that N is the same for every posterior."""

FEATURES: tuple[str, ...] = ("load", "voltage")
"""Input channels of every window."""

TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"


@dataclasses.dataclass
class PreparedData:
    """Windowed, scaled train and validation data.

    Attributes:
        train: Training windows and targets.
        val: Validation windows and targets.
    """

    train: ArrayDataSource
    val: ArrayDataSource


def _split_train_val(
    train_path: pathlib.Path, n_histories_train: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training CSV into training and validation histories by ``run_id``.

    Args:
        train_path: Path of ``train.csv``.
        n_histories_train: Histories with a smaller ``run_id`` are training data.

    Returns:
        The training and the validation histories.
    """
    data = pd.read_csv(train_path)
    return (
        data[data["run_id"] < n_histories_train],
        data[data["run_id"] >= n_histories_train],
    )


def _shuffled_batches(
    features: np.ndarray, targets: np.ndarray, batch_size: int, seed: int
) -> list[Batch]:
    """Split one epoch's shuffled training windows into batches.

    Args:
        features: Training windows, one per row.
        targets: Training targets, one per row.
        batch_size: Batch size (the last batch may be smaller).
        seed: Seed of this epoch's shuffle.

    Returns:
        The batches, every window appearing exactly once.
    """
    order = np.random.default_rng(seed).permutation(len(targets))
    return [
        Batch(inputs=jnp.asarray(features[idx]), targets=jnp.asarray(targets[idx]))
        for idx in np.split(order, range(batch_size, len(order), batch_size))
    ]


def make_data_pipeline(
    window_size: int = WINDOW_SIZE,
    x_scaler: DataScaler | None = None,
    y_scaler: DataScaler | None = None,
) -> DataPipeline:
    """Build the multiple-scenarios data pipeline (scaling, then windowing).

    Args:
        window_size: Number of time steps per window (stride 1).
        x_scaler: Scaler of the input features. Defaults to a ``StandardScaler``.
        y_scaler: Scaler of the RUL target. Defaults to a ``MinMaxScaler`` on
            ``(0, 1)``.

    Returns:
        A pipeline mapping a histories dataframe to ``(X (N, W, C), y (N, 1))``.
    """
    return DataPipeline(
        [
            ScalingStep(
                scaler=x_scaler if x_scaler is not None else skpp.StandardScaler(),
                features=list(FEATURES),
            ),
            ScalingStep(
                scaler=(
                    y_scaler
                    if y_scaler is not None
                    else skpp.MinMaxScaler(feature_range=(0, 1))
                ),
                features=["rul"],
            ),
            functools.partial(
                get_time_windows_and_join,
                window_size=window_size,
                stride=1,
                features=list(FEATURES),
            ),
            add_feature_dimension_to_y,
            to_jax,
        ]
    )


def prepare_windows(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pipeline: DataPipeline,
    *,
    fit_scalers: bool = True,
) -> PreparedData:
    """Window the train and validation histories with train-fitted scalers.

    Args:
        train_df: Training histories.
        val_df: Validation histories.
        pipeline: The data pipeline.
        fit_scalers: Fit the pipeline's scalers on ``train_df`` (a backbone run).
            ``False`` uses already fitted scalers as they are (runs consuming a
            backbone, see ``backbone_data_pipeline``).

    Returns:
        The prepared train and validation windows.
    """
    pipeline.set_mode(
        ScalingMode.FIT_TRANSFORM if fit_scalers else ScalingMode.TRANSFORM
    )
    X_train, y_train = pipeline(train_df)

    pipeline.set_mode(ScalingMode.TRANSFORM)
    X_val, y_val = pipeline(val_df)

    return PreparedData(
        train=ArrayDataSource(features=X_train, targets=y_train),
        val=ArrayDataSource(features=X_val, targets=y_val),
    )


def posterior_dataset_size(data: PreparedData) -> int:
    """Return the posterior's dataset size N: the number of training windows.

    Args:
        data: The prepared windows.

    Returns:
        The number of training windows (stride 1).
    """
    return len(data.train)


def _shuffle_seed(key: jax.Array, epoch: int) -> int:
    """Derive the integer shuffle seed of one epoch from the loader key.

    Args:
        key: The loader's typed PRNG key.
        epoch: The epoch index.

    Returns:
        A non-negative integer seed, the same for the same ``(key, epoch)``.
    """
    epoch_key = jax.random.fold_in(key, epoch)
    return int(jax.random.randint(epoch_key, (), 0, jnp.iinfo(jnp.int32).max))


def make_loader(data: PreparedData, batch_size: int, key: jax.Array) -> LoaderFactory:
    """Build stagereg's ``LoaderFactory`` over the prepared windows.

    The training split is reshuffled every epoch with a seed derived from
    ``key`` and the epoch, so the same epoch always gives the same batch order.
    The validation split is one batch holding every validation window, in order.

    Args:
        data: The prepared windows.
        batch_size: Training batch size (the last batch may be smaller).
        key: Typed PRNG key the training shuffle derives from.

    Returns:
        A loader ``loader(*, split, epoch) -> list[Batch]``.
    """
    # Converted once: indexing NumPy is far cheaper than indexing device arrays.
    train_features = np.asarray(data.train.features)
    train_targets = np.asarray(data.train.targets)
    val_batch = Batch(
        inputs=jnp.asarray(data.val.features), targets=jnp.asarray(data.val.targets)
    )

    def loader(*, split: str, epoch: int) -> list[Batch]:
        """Return the batches of one split and epoch.

        Args:
            split: ``"train"`` or ``"validation"``.
            epoch: The epoch index.

        Returns:
            The batches.

        Raises:
            ValueError: If the split is unknown.
        """
        if split == TRAIN_SPLIT:
            return _shuffled_batches(
                train_features, train_targets, batch_size, _shuffle_seed(key, epoch)
            )
        if split == VALIDATION_SPLIT:
            return [val_batch]
        raise ValueError(
            f"Unknown split {split!r}; expected {TRAIN_SPLIT!r} or "
            f"{VALIDATION_SPLIT!r}."
        )

    return loader


def restore_only_loader(*, split: str, epoch: int) -> Iterable[Batch]:
    """The loader of a stage that is only restored, never trained.

    Args:
        split: The requested split (never served).
        epoch: The requested epoch (never served).

    Raises:
        RuntimeError: Always; a restored stage must not read data.
    """
    raise RuntimeError(
        f"A restored stage must not read data (asked for {split!r}, epoch {epoch})."
    )


def prepare_data(
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    pipeline: DataPipeline,
    *,
    fit_scalers: bool = True,
) -> PreparedData:
    """Prepare the multiple-scenarios windows from the raw training histories.

    Histories with ``run_id < n_histories_train`` (read from the data-generation
    run's MLflow params) are training data, the rest are validation data.

    Args:
        raw_data_dir: Directory holding ``train.csv``.
        data_gen_run_id: MLflow run ID of the data-generation run. The MLflow
            tracking URI must already be set.
        pipeline: The data pipeline (see ``make_data_pipeline``).
        fit_scalers: Fit the pipeline's scalers on the training histories;
            ``False`` keeps already fitted ones (see ``prepare_windows``).

    Returns:
        The prepared train and validation windows.
    """
    data_gen_run = mlflow.get_run(data_gen_run_id)
    train_df, val_df = _split_train_val(
        raw_data_dir / "train.csv",
        n_histories_train=int(data_gen_run.data.params["n_histories_train"]),
    )

    return prepare_windows(train_df, val_df, pipeline, fit_scalers=fit_scalers)
