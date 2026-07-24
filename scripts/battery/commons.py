from __future__ import annotations

import dataclasses
import pathlib
from typing import Iterator

import grain
import numpy as np
import pandas as pd
import simbat as sb

from qmodem.data import DataSource

DATA_GEN_RUN_ID = "48ce4a61104840c58e892006c1bc7880"

BATTERY_DATA_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "battery"
)


@dataclasses.dataclass
class TrainHyperparameters:
    conv_kernel_size: int = 5
    conv_n_filters: int = 4
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
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    scheduler_alpha: float = 0.1
    n_samples_predictive_mean_variance: int = 100
    activation_function: str = "gelu"


def get_dataframes(
    train_path: pathlib.Path, test_path: pathlib.Path, n_histories_train: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    # Split the train dataframe: if the run ID is < 100, then it goes in the training set, otherwise in the validation set. This way we ensure that the same RNG seed will always produce the same split.
    train_df, val_df = (
        train_df[train_df["run_id"] < n_histories_train],
        train_df[train_df["run_id"] >= n_histories_train],
    )
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df


def train_dataloader_builder(
    sampler_seed: int,
    ds_train: DataSource,
    batch_size: int,
    drop_remainder: bool = False,
) -> grain.DataLoader:
    """Create Grain DataLoader for training."""

    sampler_train = grain.samplers.IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=True, seed=sampler_seed
    )
    dataloader_train = grain.DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[
            grain.transforms.Batch(batch_size=batch_size, drop_remainder=drop_remainder)
        ],
        worker_count=0,
    )

    return dataloader_train


def run_discharges_from_intermediate_socs(
    soc_0s: np.ndarray, process_noise_std: float, dt: float
) -> Iterator[sb.SimulationResult]:
    for soc_0 in soc_0s:
        # TODO: simulation config parameters should be loaded from mlflow.
        config = sb.SimulationConfig(
            process_noise_distribution=lambda: np.random.normal(
                loc=0.0, scale=process_noise_std
            ),
            measurement_noise_distribution=lambda: 0.0,
            dt=dt,
            soc_0=soc_0,
        )
        result = sb.simulate_constant_capacity_simple(n_sim=100, config=config)
        yield result
