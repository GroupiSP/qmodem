from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import grain
import pandas as pd

from qmodem.data import DataFrameSource, DataSource, make_battery_data_pipeline

from .hnn_model import Net


@dataclass
class Hyperparameters:
    # TODO: this should become shared among the battery scripts.
    batch_size: int = 32
    window_size: int = 20
    stride: int = 1
    normalize_rul: bool = True
    sampler_seeds: tuple[int, int] = (42, 0)
    net_init_seed: int = 0
    drop_remainder: bool = False


def get_dataframes(
    train_path: pathlib.Path, test_path: pathlib.Path
) -> tuple[pd.DataFrame, pd.Dataframe, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    # Split the train dataframe: if the run ID is < 100, then it goes in the training set, otherwise in the validation set. This way we ensure that the same RNG seed will always produce the same split.
    train_df, val_df = (
        train_df[train_df["run_id"] < 100],
        train_df[train_df["run_id"] >= 100],
    )
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def create_dataloaders(
    ds_train: DataSource,
    ds_val: DataSource,
    batch_size: int,
    sampler_seeds: tuple[int, int],
    drop_remainder: bool = False,
) -> tuple[Any, Any]:
    """Create Grain DataLoaders for training and validation."""
    from grain.transforms import Batch

    sampler_train = grain.samplers.IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=True, seed=sampler_seeds[0]
    )
    dataloader_train = grain.DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
        worker_count=0,
    )

    sampler_val = grain.samplers.IndexSampler(
        num_records=len(ds_val), num_epochs=1, shuffle=False, seed=sampler_seeds[1]
    )
    dataloader_val = grain.DataLoader(
        data_source=ds_val,
        sampler=sampler_val,
        operations=[Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
        worker_count=0,
    )

    return dataloader_train, dataloader_val


def main() -> None:
    hp = Hyperparameters()

    RAW_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    train_path, test_path = RAW_DATA_DIR / "train.csv", RAW_DATA_DIR / "test.csv"

    # Build the data sources, including windowing and normalization
    data_pipeline = make_battery_data_pipeline(
        window_size=hp.window_size, stride=hp.stride, normalize=hp.normalize_rul
    )

    train_df, val_df, test_df = get_dataframes(train_path, test_path)

    ds_train = DataFrameSource(df=train_df, pipeline=data_pipeline)
    ds_val = DataFrameSource(df=val_df, pipeline=data_pipeline)
    # ds_test = DataFrameSource(df=test_df, pipeline=data_pipeline)

    # Dataloaders
    create_dataloaders(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=hp.batch_size,
        sampler_seeds=hp.sampler_seeds,
        drop_remainder=hp.drop_remainder,
    )

    # Model
    Net()


if __name__ == "__main__":
    main()
