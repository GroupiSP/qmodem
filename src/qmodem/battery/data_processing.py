from __future__ import annotations

import dataclasses
import functools
import pathlib
from collections.abc import Callable, Iterable

import grain
import jax
import jax.numpy as jnp
import mlflow
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
    train: ArrayDataSource
    val: ArrayDataSource


def _split_train_val(
    train_path: pathlib.Path, n_histories_train: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(train_path)
    return (
        data[data["run_id"] < n_histories_train],
        data[data["run_id"] >= n_histories_train],
    )


def _train_dataloader_builder(
    sampler_seed: int,
    ds_train: ArrayDataSource,
    batch_size: int,
    drop_remainder: bool,
) -> grain.DataLoader:
    sampler = grain.samplers.IndexSampler(
        num_records=len(ds_train),
        num_epochs=1,
        shuffle=True,
        seed=sampler_seed,
    )
    return grain.DataLoader(
        data_source=ds_train,
        sampler=sampler,
        operations=[
            grain.transforms.Batch(
                batch_size=batch_size,
                drop_remainder=drop_remainder,
            )
        ],
        worker_count=0,
    )


def dataloader_builders(
    data: PreparedData, batch_size: int, drop_remainder: bool
) -> tuple[Callable[[int], Iterable], Callable[[int], Iterable]]:
    train_builder = functools.partial(
        _train_dataloader_builder,
        ds_train=data.train,
        batch_size=batch_size,
        drop_remainder=drop_remainder,
    )

    def val_builder(epoch: int) -> list[tuple[jax.Array, jax.Array]]:
        """The validation dataloader is a convention for consistency with the training
        loop.

        It returns a single batch containing the entire validation set, which is assumed
        to be small enough to fit in memory.
        """
        return [(data.val.features, data.val.targets)]

    return train_builder, val_builder


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
    train_df: pd.DataFrame, val_df: pd.DataFrame, pipeline: DataPipeline
) -> PreparedData:
    """Window the train and validation histories with train-fitted scalers.

    Args:
        train_df: Training histories.
        val_df: Validation histories.
        pipeline: The data pipeline. Its scalers are fitted on ``train_df``.

    Returns:
        The prepared train and validation windows.
    """
    pipeline.set_mode(ScalingMode.FIT_TRANSFORM)
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
    """Derive the integer sampler seed of one epoch from the loader key.

    Args:
        key: The loader's typed PRNG key.
        epoch: The epoch index.

    Returns:
        A non-negative integer seed, the same for the same ``(key, epoch)``.
    """
    epoch_key = jax.random.fold_in(key, epoch)
    return int(jax.random.randint(epoch_key, (), 0, jnp.iinfo(jnp.int32).max))


def make_loader(data: PreparedData, batch_size: int, key: jax.Array) -> LoaderFactory:
    """Adapt the dataloader builders to stagereg's ``LoaderFactory``.

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
    train_builder, val_builder = dataloader_builders(
        data, batch_size=batch_size, drop_remainder=False
    )

    def loader(*, split: str, epoch: int) -> list[Batch]:
        if split == TRAIN_SPLIT:
            batches = train_builder(_shuffle_seed(key, epoch))
        elif split == VALIDATION_SPLIT:
            batches = val_builder(epoch)
        else:
            raise ValueError(
                f"Unknown split {split!r}; expected {TRAIN_SPLIT!r} or "
                f"{VALIDATION_SPLIT!r}."
            )
        return [
            Batch(inputs=jnp.asarray(x), targets=jnp.asarray(y)) for x, y in batches
        ]

    return loader


def prepare_data(
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    pipeline: DataPipeline,
) -> PreparedData:
    """Prepare the multiple-scenarios windows from the raw training histories.

    Histories with ``run_id < n_histories_train`` (read from the data-generation
    run's MLflow params) are training data, the rest are validation data.

    Args:
        raw_data_dir: Directory holding ``train.csv``.
        data_gen_run_id: MLflow run ID of the data-generation run. The MLflow
            tracking URI must already be set.
        pipeline: The data pipeline (see ``make_data_pipeline``).

    Returns:
        The prepared train and validation windows.
    """
    data_gen_run = mlflow.get_run(data_gen_run_id)
    train_df, val_df = _split_train_val(
        raw_data_dir / "train.csv",
        n_histories_train=int(data_gen_run.data.params["n_histories_train"]),
    )

    return prepare_windows(train_df, val_df, pipeline)
