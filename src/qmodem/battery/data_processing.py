from __future__ import annotations

import dataclasses
import functools
import pathlib
from typing import Callable, Iterable

import grain
import jax
import mlflow
import numpy as np
import pandas as pd
import sklearn.preprocessing as skpp

from qmodem.data import (
    ArrayDataSource,
    DataPipeline,
    DataScaler,
    IdentityScaler,
    ScalingMode,
    ScalingStep,
    add_feature_dimension_to_y,
    get_time_windows_and_join,
    to_jax,
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


class _ScalerWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, scaler: DataScaler, predict_method: str = "transform"):
        self.scaler = scaler
        self.predict_method = predict_method

    def predict(self, model_input: np.ndarray, params=None):
        return getattr(self.scaler, self.predict_method)(model_input)


def mlflow_log_scaler(
    scaler: DataScaler, name: str, predict_method: str = "transform"
) -> None:
    """Wrapper around the mlflow sklearn model logging functionality. If the scaler is
    not an sklearn estimator, it does not log anything.

    Args:
        scaler: The scaler to log.
        name: The name of the MLFlow model for the scaler.
        predict_method: The method of the scaler to use for prediction (e.g., "inverse_transform").
    """
    mlflow.pyfunc.log_model(
        name=name,
        python_model=_ScalerWrapper(scaler, predict_method=predict_method),
        input_example=np.array([[0.0], [1.0]], dtype=np.float32),
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
    window_size: int,
    stride: int,
    normalize_rul: bool,
) -> PreparedData:
    data_gen_run = mlflow.get_run(data_gen_run_id)
    train_df, val_df = _split_train_val(
        raw_data_dir / "train.csv",
        n_histories_train=int(data_gen_run.data.params["n_histories_train"]),
    )

    # TODO Pass pipeline as an argument, rather than creating them here.
    rul_scaler = (
        skpp.MinMaxScaler(feature_range=(0, 1)) if normalize_rul else IdentityScaler()
    )

    pipeline = DataPipeline(
        [
            functools.partial(
                get_time_windows_and_join,
                window_size=window_size,
                stride=stride,
                features=["voltage"],
            ),
            add_feature_dimension_to_y,
            ScalingStep(x_scaler=IdentityScaler(), y_scaler=rul_scaler),
            to_jax,
        ]
    )

    # Apply the pipeline to the training data, then set the mode to TRANSFORM for the validation data.
    # This ensures that the scalers are fitted on the training data only.
    pipeline.set_mode(ScalingMode.FIT_TRANSFORM)
    X_train, y_train = pipeline(train_df)

    pipeline.set_mode(ScalingMode.TRANSFORM)
    X_val, y_val = pipeline(val_df)

    # Log the scaler with MLFlow for reproducibility and loading at test time.
    mlflow_log_scaler(rul_scaler, name="rul_scaler", predict_method="inverse_transform")

    return PreparedData(
        train=ArrayDataSource(features=X_train, targets=y_train),
        val=ArrayDataSource(features=X_val, targets=y_val),
    )
