from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any

import flax.nnx as nnx
import grain
import jax
import jax.numpy as jnp
import optax
import pandas as pd

from qmodem.data import DataFrameSource, DataSource, make_battery_data_pipeline
from qmodem.module import negative_log_likelihood
from qmodem.utils import count_parameters

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
    train_rng_seed: int = 1
    drop_remainder: bool = False
    learning_rate: float = 1e-2
    n_epochs: int = 500
    beta_nll: float = 0.5
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4


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
    dataloader_train, dataloader_val = create_dataloaders(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=hp.batch_size,
        sampler_seeds=hp.sampler_seeds,
        drop_remainder=hp.drop_remainder,
    )

    # Model, schedule, optimizer
    model = Net(rngs=nnx.Rngs(hp.net_init_seed))
    n_params = count_parameters(model)  # TODO: track
    print(f"Model has {n_params} parameters.")  # TODO: remove

    schedule = optax.cosine_decay_schedule(
        init_value=hp.learning_rate,
        decay_steps=hp.n_epochs * (len(ds_train) // hp.batch_size),
        alpha=0.1,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    @nnx.jit
    def train_step(
        model: nnx.Module,
        batch: jax.Array,
        key: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        # Split the keys for the batch
        keys = jax.random.split(key, batch[0].shape[0])

        @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
        def per_sample_nll(model, sample, sample_key):
            # Build the RNG here to avoid crossing different trace levels.
            rngs = nnx.Rngs(dropout=sample_key)
            return negative_log_likelihood(model, sample, rngs, beta=hp.beta_nll)

        def loss_fn(model):
            return jnp.mean(per_sample_nll(model, batch, keys))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    # Do a single train step for debugging purposes
    batch = next(iter(dataloader_train))
    key = jax.random.PRNGKey(hp.train_rng_seed)
    loss = train_step(model, batch, key, optimizer)
    print(f"Train step loss: {loss:.6f}")

    # @nnx.jit
    # def eval_step(
    #     model: nnx.Module,
    #     batch: jax.Array,
    #     rngs: nnx.Rngs,
    # ) -> jax.Array:
    #     return nll_loss(model, batch, rngs, hp.beta_nll)

    # early_stopper = EarlyStopper(
    #     patience=hp.early_stopping_patience, min_delta=hp.early_stopping_min_delta
    # )
    # report_condition = ReportConditionEvery(hp.print_every)
    # reporter = train_report_print

    # # TODO: handle RNGs in the training loop
    # # TODO: track
    # best_val_loss, _ = train_loop(
    #     n_epochs=hp.n_epochs,
    #     dataloader_train=dataloader_train,
    #     dataloader_val=dataloader_val,
    #     train_batch_fn=lambda batch: train_step(model, optimizer, batch),
    #     eval_batch_fn=lambda batch: eval_step(model, batch),
    #     early_stopper=early_stopper,
    #     report_condition=report_condition,
    #     reporter=reporter,
    #     on_train_epoch_start=model.train,
    #     on_val_epoch_start=model.eval,
    # )


if __name__ == "__main__":
    main()
