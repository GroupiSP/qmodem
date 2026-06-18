from __future__ import annotations

import numpy as np

from qmodem.data import CMAPSSAnalyst, prepare_cmapss, split_cmapss
from qmodem.utils import CMAPSS_DIR_PATH, PROCESSED_DATA_DIR_PATH


def main() -> None:
    SEED = 42
    NUM_SENSORS_RETAINED = 9
    RELATIVE_SUBSET_SIZE = 0.2

    np.random.seed(SEED)

    data_path = CMAPSS_DIR_PATH / "train_FD001.txt"
    df = prepare_cmapss(data_path)

    analyst = CMAPSSAnalyst(df)

    train_df, test_df = split_cmapss(df, relative_subset_size=RELATIVE_SUBSET_SIZE)
    train_df, val_df = split_cmapss(train_df, relative_subset_size=RELATIVE_SUBSET_SIZE)

    metrics_cmapss = analyst.compute_prognostic_metrics(train_df)
    metrics_cmapss.to_csv(
        PROCESSED_DATA_DIR_PATH / "metrics_cmapss_fd001_train.csv", index=False
    )

    sensors_selected = metrics_cmapss.head(NUM_SENSORS_RETAINED)["sensor_name"].tolist()

    # filter the dataframes to only include the selected sensors (exclude also the operational settings)
    non_sensor_columns = [
        col
        for col in df.columns
        if not (col.startswith("sensor_") or col.startswith("op_setting_"))
    ]
    columns_selected = non_sensor_columns + sensors_selected

    train_df = train_df[columns_selected]
    val_df = val_df[columns_selected]
    test_df = test_df[columns_selected]

    # save the filtered dataframes to csv
    train_df.to_csv(
        PROCESSED_DATA_DIR_PATH / "cmapss_fd001_train_train.csv", index=False
    )
    val_df.to_csv(PROCESSED_DATA_DIR_PATH / "cmapss_fd001_train_val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR_PATH / "cmapss_fd001_train_test.csv", index=False)


if __name__ == "__main__":
    main()
