from __future__ import annotations

import pathlib
from typing import Sequence

from qmodem.data import DataFrameSource, make_battery_data_pipeline


def get_data_paths(data_dir: pathlib.Path) -> tuple[Sequence[pathlib.Path]]:
    all_tv_paths = sorted(data_dir.glob("train_history_[0-9]*.csv"))
    training_paths = [p for p in all_tv_paths if int(p.stem.split("_")[-1]) < 100]
    validation_paths = [p for p in all_tv_paths if int(p.stem.split("_")[-1]) >= 100]
    test_paths = sorted(data_dir.glob("test_history_[0-9]*.csv"))
    return training_paths, validation_paths, test_paths


def main() -> None:
    RAW_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    training_paths, validation_paths, test_paths = get_data_paths(RAW_DATA_DIR)

    # Build the data sources, including windowing and normalization
    data_pipeline = make_battery_data_pipeline(window_size=20, stride=1, normalize=True)

    DataFrameSource(paths=list(training_paths), pipeline=data_pipeline)
    DataFrameSource(paths=list(validation_paths), pipeline=data_pipeline)
    DataFrameSource(paths=list(test_paths), pipeline=data_pipeline)


if __name__ == "__main__":
    main()
