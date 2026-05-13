from __future__ import annotations

import pathlib

from qmodem.data import DataFrameSource, make_battery_data_pipeline


def main() -> None:
    RAW_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    # Process training data
    training_paths = RAW_DATA_DIR.glob("train_history_[0-9]*.csv")

    DataFrameSource(paths=list(training_paths), pipeline=make_battery_data_pipeline())


if __name__ == "__main__":
    main()
