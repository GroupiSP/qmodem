from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Callable, Iterable

import grain
import jax
import mlflow
import pandas as pd
import sklearn.preprocessing as skpp

from qmodem.data import (
    DataFrameSource,
    DataPipeline,
    DataSource,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    normalize_ruls,
    to_jax,
)


@dataclasses.dataclass
class PreparedData:
    train: DataFrameSource
    val: DataFrameSource
    scaler: skpp.MinMaxScaler


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
    ds_train: DataSource,
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
        return [(data.val.X, data.val.y)]

    return train_builder, val_builder


def prepare_data(
    raw_data_dir: pathlib.Path,
    data_gen_run_id: str,
    window_size: int,
    stride: int,
    normalize_rul: bool,
) -> PreparedData:
    data_gen_run = mlflow.get_run(data_gen_run_id)
    train_df, val_df = _split_train_val(
        raw_data_dir / "train.csv",
        n_histories_train=int(data_gen_run.data.params["n_histories_train"]),
    )

    scaler = skpp.MinMaxScaler(feature_range=(0, 1))
    common_steps = [
        functools.partial(
            get_time_windows_and_join,
            window_size=window_size,
            stride=stride,
        ),
        add_feature_dimension_to_y,
    ]
    train_pipeline = DataPipeline(
        [
            *common_steps,
            functools.partial(normalize_ruls, transform_fn=scaler.fit_transform)
            if normalize_rul
            else lambda data: data,
            to_jax,
        ]
    )
    val_pipeline = DataPipeline(
        [
            *common_steps,
            functools.partial(normalize_ruls, transform_fn=scaler.transform)
            if normalize_rul
            else lambda data: data,
            to_jax,
        ]
    )

    return PreparedData(
        train=DataFrameSource(df=train_df, pipeline=train_pipeline),
        val=DataFrameSource(df=val_df, pipeline=val_pipeline),
        scaler=scaler,
    )
