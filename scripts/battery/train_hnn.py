from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import grain
import jax
import jax.numpy as jnp
import pandas as pd

from qmodem.data import DataFrameSource, DataSource, make_battery_data_pipeline
from qmodem.module import GaussianBlock


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


class Net(nnx.Module):
    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 5,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Heteroscedastic 1D CNN for time-series RUL prediction with uncertainty.

        Architecture: Conv1D -> Activation -> Global Average Pooling -> GaussianBlock
        Outputs both mean and variance predictions. Accepts variable-length input
        windows.

        Args:
            n_filters (int, optional): Number of convolutional filters. Defaults to 4.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 5.
            act_fn (nnx.Module, optional): Activation function. Defaults to nnx.gelu.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.act_fn = act_fn

        self.conv = nnx.Conv(
            in_features=1,
            out_features=n_filters,
            kernel_size=(kernel_size,),
            padding="VALID",
            rngs=rngs,
        )

        # GaussianBlock to output mean and variance
        self.gauss = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the heteroscedastic CNN.

        Args:
            x (jax.Array): Input with shape (batch, window_size, 1).
                           Accepts variable-length windows.

        Returns:
            jax.Array: Concatenated [mu, var_positive] with shape (batch, 2).
        """
        # Conv1D with activation
        x = self.conv(x)
        x = self.act_fn(x)

        # Global Average Pooling: (batch, window_size, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=1)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gauss(x)


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


if __name__ == "__main__":
    main()
