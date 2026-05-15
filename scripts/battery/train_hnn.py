from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Any, Sequence

import grain

from qmodem.data import DataFrameSource, DataSource, make_battery_data_pipeline


@dataclass
class Hyperparameters:
    # TODO: this should become shared among the battery scripts.
    batch_size: int
    window_size: int
    stride: int
    normalize_rul: bool
    sampler_seeds: tuple[int, int]
    drop_remainder: bool


def get_data_paths(data_dir: pathlib.Path) -> tuple[Sequence[pathlib.Path]]:
    all_tv_paths = sorted(data_dir.glob("train_history_[0-9]*.csv"))
    training_paths = [p for p in all_tv_paths if int(p.stem.split("_")[-1]) < 100]
    validation_paths = [p for p in all_tv_paths if int(p.stem.split("_")[-1]) >= 100]
    test_paths = sorted(data_dir.glob("test_history_[0-9]*.csv"))
    return training_paths, validation_paths, test_paths


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
    hp = Hyperparameters(
        batch_size=32,
        window_size=20,
        stride=1,
        normalize_rul=True,
        sampler_seeds=(42, 0),
        drop_remainder=False,
    )

    RAW_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    training_paths, validation_paths, test_paths = get_data_paths(RAW_DATA_DIR)

    # Build the data sources, including windowing and normalization
    data_pipeline = make_battery_data_pipeline(
        window_size=hp.window_size, stride=hp.stride, normalize=hp.normalize_rul
    )

    ds_train = DataFrameSource(paths=list(training_paths), pipeline=data_pipeline)
    ds_val = DataFrameSource(paths=list(validation_paths), pipeline=data_pipeline)
    DataFrameSource(paths=list(test_paths), pipeline=data_pipeline)

    # Dataloaders
    create_dataloaders(
        ds_train=ds_train,
        ds_val=ds_val,
        batch_size=hp.batch_size,
        sampler_seeds=hp.sampler_seeds,
        drop_remainder=hp.drop_remainder,
    )


if __name__ == "__main__":
    main()
