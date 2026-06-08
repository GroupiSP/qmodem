from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import grain
import pandas as pd

from qmodem.data import DataSource

DATA_GEN_RUN_ID = "db5ed872d78448dba887040cf49a74d1"


@dataclasses.dataclass
class TrainHyperparameters:
    batch_size: int = 32
    window_size: int = 20
    stride: int = 1
    normalize_rul: bool = True
    sampler_seeds: tuple[int, int] = (42, 0)
    net_init_seed: int = 0
    train_rng_seed: int = 1
    drop_remainder: bool = False
    learning_rate: float = 1e-2
    n_epochs: int = 500
    beta_nll: float = 0.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    scheduler_alpha: float = 0.1


def get_dataframes(
    train_path: pathlib.Path, test_path: pathlib.Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    sampler_train = grain.samplers.IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=True, seed=sampler_seeds[0]
    )
    dataloader_train = grain.DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[
            grain.transforms.Batch(batch_size=batch_size, drop_remainder=drop_remainder)
        ],
        worker_count=0,
    )

    sampler_val = grain.samplers.IndexSampler(
        num_records=len(ds_val), num_epochs=1, shuffle=False, seed=sampler_seeds[1]
    )
    dataloader_val = grain.DataLoader(
        data_source=ds_val,
        sampler=sampler_val,
        operations=[
            grain.transforms.Batch(batch_size=batch_size, drop_remainder=drop_remainder)
        ],
        worker_count=0,
    )

    return dataloader_train, dataloader_val
