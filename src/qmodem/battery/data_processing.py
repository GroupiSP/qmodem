from __future__ import annotations

import dataclasses
import functools
import pathlib
from collections.abc import Callable, Iterable

import grain
import jax
import mlflow
import pandas as pd

from qmodem.data import (
    ArrayDataSource,
    DataPipeline,
    ScalingMode,
)


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


def prepare_data(
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    pipeline: DataPipeline,
) -> PreparedData:
    data_gen_run = mlflow.get_run(data_gen_run_id)
    train_df, val_df = _split_train_val(
        raw_data_dir / "train.csv",
        n_histories_train=int(data_gen_run.data.params["n_histories_train"]),
    )

    # TODO Pass pipeline as an argument, rather than creating them here.
    # Apply the pipeline to the training data, then set the mode to TRANSFORM for the validation data.
    # This ensures that the scalers are fitted on the training data only.
    pipeline.set_mode(ScalingMode.FIT_TRANSFORM)
    X_train, y_train = pipeline(train_df)

    pipeline.set_mode(ScalingMode.TRANSFORM)
    X_val, y_val = pipeline(val_df)

    return PreparedData(
        train=ArrayDataSource(features=X_train, targets=y_train),
        val=ArrayDataSource(features=X_val, targets=y_val),
    )
