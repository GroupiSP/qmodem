from __future__ import annotations

import pathlib
from enum import StrEnum, auto
from typing import Callable, Protocol, Sequence, SupportsIndex

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def _make_windows(
    features: np.ndarray,
    ruls: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[list[np.ndarray], list[float]]:
    """Extract sliding time windows and corresponding RUL targets from a single
    discharge history.

    If the history is shorter than *window_size*, the voltage and RUL arrays are
    left-edge-padded (the first value is repeated) so that at least one window
    is produced.

    Args:
        features: N_i-D feature history of shape ``(N_t, N_i)``.
        ruls: 1-D RUL values of shape ``(N_t,)`` aligned with *features*.
        window_size: Number of time steps per window.
        stride: Step size for the sliding window.

    Returns:
        A tuple ``(windows, targets)`` where each window has shape
        ``(window_size, N_i)`` and each target is a scalar RUL value.
    """
    N_t = features.shape[0]

    # Left-edge-pad short histories so at least one full window can be made.
    if N_t < window_size:
        pad_len = window_size - N_t
        features = np.concatenate(
            [
                np.full(shape=(pad_len, features.shape[1]), fill_value=features[0]),
                features,
            ]
        )
        ruls = np.concatenate([np.full(pad_len, ruls[0]), ruls])
        N_t = window_size

    windows: list[np.ndarray] = []
    targets: list[float] = []

    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        windows.append(features[start:end])
        targets.append(float(ruls[end]))

    windows.append(features[-window_size:])
    targets.append(0.0)

    return windows, targets


class DataSource(Protocol):
    """Protocol for data sources that can be used with Grain DataLoaders."""

    def __len__(self) -> int:
        """Returns the number of records in the dataset."""
        ...

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, jax.Array]:
        """Retrieves the features and target for the given record key.

        Args:
            record_key (SupportsIndex): An index or slice to specify which record(s) to retrieve.
        Returns:
            tuple[jax.Array, jax.Array]: A tuple of (features, target).
                - features: A jax.Array containing the input features for the specified record(s).
                - target: A jax.Array containing the target values for the specified record(s).
        """
        ...


class DataScaler(Protocol):
    def fit(self, x: jaxtyping.ArrayLike, y: jaxtyping.ArrayLike) -> None: ...
    def fit_transform(
        self, x: jaxtyping.ArrayLike, y: jaxtyping.ArrayLike
    ) -> tuple[jaxtyping.ArrayLike, jaxtyping.ArrayLike]: ...
    def transform(self, x: jaxtyping.ArrayLike) -> jaxtyping.ArrayLike: ...
    def inverse_transform(self, x: jaxtyping.ArrayLike) -> jaxtyping.ArrayLike: ...


class IdentityScaler:
    """A scaler that performs no scaling, returning the input as-is."""

    def fit(self, x: jaxtyping.ArrayLike, y: jaxtyping.ArrayLike) -> None:
        """No-op for fitting the scaler."""
        pass

    def fit_transform(
        self, x: jaxtyping.ArrayLike, y: jaxtyping.ArrayLike = None
    ) -> tuple[jaxtyping.ArrayLike, jaxtyping.ArrayLike]:
        """Returns the input arrays as-is without any scaling."""
        return x

    def transform(self, x: jaxtyping.ArrayLike) -> jaxtyping.ArrayLike:
        """Returns the input array as-is without any scaling."""
        return x

    def inverse_transform(self, x: jaxtyping.ArrayLike) -> jaxtyping.ArrayLike:
        """Returns the input array as-is without any scaling."""
        return x


class ArrayDataSource:
    """A simple implementation of the DataSource protocol that wraps feature and target
    arrays."""

    def __init__(
        self, features: jaxtyping.ArrayLike, targets: jaxtyping.ArrayLike
    ) -> None:
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(
        self, record_key: SupportsIndex
    ) -> tuple[jaxtyping.ArrayLike, jaxtyping.ArrayLike]:
        return self.features[record_key], self.targets[record_key]


class ScalingMode(StrEnum):
    FIT_TRANSFORM = auto()
    TRANSFORM = auto()


class ScalingStep:
    """A data processing step that scales features and targets using provided scalers.

    Meant to be used in a data pipeline.
    """

    @staticmethod
    def _identity_transform(x: jaxtyping.ArrayLike) -> jaxtyping.ArrayLike:
        return x

    def __init__(
        self,
        x_scaler: DataScaler = IdentityScaler(),
        y_scaler: DataScaler = IdentityScaler(),
    ) -> None:
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self._mode = ScalingMode.FIT_TRANSFORM

    @property
    def mode(self) -> ScalingMode:
        return self._mode

    @mode.setter
    def mode(self, value: ScalingMode) -> None:
        if not isinstance(value, ScalingMode):
            raise ValueError(f"mode must be an instance of ScalingMode, got {value}")
        self._mode = value

    def _fit_transform(self, x, y):
        x_scaled = self.x_scaler.fit_transform(x)
        y_scaled = self.y_scaler.fit_transform(y)
        return x_scaled, y_scaled

    def _transform(self, x, y):
        return self.x_scaler.transform(x), self.y_scaler.transform(y)

    _dispatch = {
        ScalingMode.FIT_TRANSFORM: _fit_transform,
        ScalingMode.TRANSFORM: _transform,
    }

    def __call__(
        self, data: tuple[jaxtyping.ArrayLike, jaxtyping.ArrayLike]
    ) -> tuple[jaxtyping.ArrayLike, jaxtyping.ArrayLike]:
        x, y = data
        return self._dispatch[self._mode](self, x, y)


class DataPipeline:
    def __init__(self, steps: Sequence[Callable]) -> None:
        self.steps = steps

    def __call__(self, x: pd.DataFrame) -> tuple[jax.Array, jax.Array]:
        for step in self.steps:
            x = step(x)
        return x

    def set_mode(self, mode: ScalingMode) -> None:
        for step in self.steps:
            if isinstance(step, ScalingStep):
                step.mode = mode


def get_time_windows_and_join(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts sliding time windows and corresponding RUL targets from a dataframe
    containing multiple discharge histories.

    Args:
        df: Dataframe containing multiple discharge histories. Each history is identified by a unique value in the "run_id" column.
        window_size: Number of time steps per window.
        stride: Step size for the sliding window.
        features: List of feature column names to include in the windows.
    Returns:
        A tuple ``(feature_windows, rul_targets)`` where shape(feature_windows) = (N_w, window_size, N_i)
        and shape(rul_targets) = (N_w,), with N_w being the total number of windows across all histories.
    """
    # TEST does it work for both 1D and multi-D features? Do the output arrays have the correct shape? (N_w, window_size, N_i) and (N_w,)
    # TEST does it work for a single unit?
    # TEST is the final number of windows correct? (N_w = sum_i ceil((N_t_i - window_size) / stride) + 1)
    feature_windows: list[np.ndarray] = []
    rul_targets: list[float] = []

    unit_ids = df["run_id"].unique()
    for unit_id in unit_ids:
        unit_df = df[df["run_id"] == unit_id].sort_values("time")
        feature_array = unit_df[features].values
        ruls = unit_df["time"].iloc[-1] - unit_df["time"].values

        fw_i, rul_i = _make_windows(feature_array, ruls, window_size, stride)
        feature_windows.extend(fw_i)
        rul_targets.extend(rul_i)

    return np.array(feature_windows), np.array(rul_targets)


def add_feature_dimension_to_y(
    x: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    feature_windows, rul_windows = x
    return feature_windows, rul_windows.reshape(-1, 1)


def to_jax(x: tuple[np.ndarray, np.ndarray]) -> tuple[jax.Array, jax.Array]:
    X, y = x
    return jnp.array(X), jnp.array(y)


def _load_cmapss_fd001_train(path: pathlib.Path) -> pd.DataFrame:
    column_names: list[str] = (
        [
            "unit_id",
            "time_cycles",
            "op_setting_1",
            "op_setting_2",
            "op_setting_3",
        ]
        + [f"sensor_{i}" for i in range(1, 22)]
        + ["RUL"]
    )
    return pd.read_csv(path, sep=r"\s+", header=None, names=column_names)


def _add_rul(df: pd.DataFrame) -> pd.DataFrame:
    # add the RUL column
    df["RUL"] = df.groupby("unit_id")["time_cycles"].transform(lambda x: x.max() - x)
    return df


def _exclude_constant_sensors(df: pd.DataFrame) -> pd.DataFrame:
    # drop the constant sensors
    df.drop(columns=[f"sensor_{i}" for i in [1, 5, 6, 10, 16, 18, 19]], inplace=True)
    return df


def prepare_cmapss(path: pathlib.Path) -> pd.DataFrame:
    """Loads the CMAPSS FD001/train dataset, adds the RUL labels and excludes the
    constant sensors."""
    df = _load_cmapss_fd001_train(path)
    df = _add_rul(df)
    df = _exclude_constant_sensors(df)
    return df


def split_cmapss(
    df: pd.DataFrame, relative_subset_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits the CMAPSS dataframe into two sub-dataframes.

    Args:
        df: The CMAPSS dataframe to split.
        relative_subset_size: The fraction of units to include in the second subset.

    Returns:
        The two sub-dataframes split from the original.
    """
    # shuffle the unit_ids (engine IDs)
    unit_ids = df["unit_id"].unique()

    # note: the sampling follows the numpy random state.
    # If reproducibility is desired, set the seed with np.random.seed()
    # before this step.
    shuffled_unit_ids = pd.Series(unit_ids).sample(frac=1).values

    # copy the dataframe to a temp variable to avoid modifying the original
    df = df.copy()
    df["unit_id"] = pd.Categorical(
        df["unit_id"], categories=shuffled_unit_ids, ordered=True
    )

    df.sort_values(by=["unit_id", "time_cycles"], inplace=True)

    num_units = df["unit_id"].nunique()

    # Note: `train` and `test` in the names are just labels for the two splits.
    num_test_units = int(num_units * relative_subset_size)
    test_unit_ids = shuffled_unit_ids[:num_test_units]
    train_unit_ids = shuffled_unit_ids[num_test_units:]

    train_df = df[df["unit_id"].isin(train_unit_ids)]
    test_df = df[df["unit_id"].isin(test_unit_ids)]

    return train_df, test_df


class CMAPSSAnalyst:
    """Loads, preprocesses and analyses the CMAPSS FD001/train dataset.

    Attributes:
        df: The dataframe loaded from the original CMAPSS FD001/train file, with the constant columns removed and the RUL column added.
        variable_sensors: The list of sensor column names that are not constant across the whole dataset.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        # Define the attributes
        self.df: pd.DataFrame = df
        self.variable_sensors: list[str] = [
            c for c in df.columns if c.startswith("sensor_")
        ]

    @staticmethod
    def _modified_mann_kendall(t: np.ndarray, y: np.ndarray) -> float:
        """Computes the modified Mann-Kendall index of a time series.

        Args:
            t (np.ndarray): time steps of the time series (assumed in ascending order)
            y (np.ndarray): values of the time series

        Returns:
            float: value of the modified Mann-Kendall index
        """
        mk = 0.0
        sum_of_distances = 0.0
        for i in range(len(t)):
            for j in range(i + 1, len(t)):
                mk += (t[j] - t[i]) * np.sign(y[j] - y[i])
                sum_of_distances += t[j] - t[i]
        if sum_of_distances == 0:  # fail safe
            return 0.0
        return mk / sum_of_distances

    def compute_monotonicity(self, df: pd.DataFrame) -> pd.Series:
        """Computes the monotonicity of each sensor in the training set.

        Returns:
            A pandas Series with sensor names as index and monotonicity values as data.
        """
        monotonicity = {}
        for sensor_name in self.variable_sensors:
            monotonicity[sensor_name] = (
                df.groupby("unit_id")
                .apply(
                    lambda x: self._modified_mann_kendall(
                        x["time_cycles"].values, x[sensor_name].values
                    )
                )
                .mean()
            )

        return pd.Series(monotonicity)

    def compute_prognosability(self, df: pd.DataFrame) -> pd.Series:
        """Computes the prognosability of each sensor in the training set.

        Returns:
            A pandas Series with sensor names as index and prognosability values as data.
        """
        lasts_df = df.groupby("unit_id")[self.variable_sensors].last()
        firsts_df = df.groupby("unit_id")[self.variable_sensors].first()

        return (lasts_df.std() / (firsts_df - lasts_df).abs().mean()).apply(
            lambda x: np.exp(-x)
        )

    def compute_trendability(self, df: pd.DataFrame) -> pd.Series:
        """Computes the trendability of each sensor in the training set.

        Returns:
            A pandas Series with sensor names as index and trendability values as data.
        """
        trendability = {}
        for sensor_name in self.variable_sensors:
            pivot_table = df.pivot(
                index="time_cycles", columns="unit_id", values=sensor_name
            )
            cov_matrix = pivot_table.cov()
            stds = pivot_table.std()
            rho_matrix = cov_matrix / (stds.values[:, None] * stds.values[None, :])
            trendability[sensor_name] = np.abs(
                rho_matrix.values[np.triu_indices_from(rho_matrix, k=1)]
            ).min()

        return pd.Series(trendability)

    def compute_prognostic_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes all the prognostic metrics (monotonicity, prognosability,
        trendability) and the sensor fitness for each sensor in the training set.

        Returns:
            A pandas DataFrame with sensor names as index and the metrics as columns. The dataframe is sorted by fitness in descending order.
        """
        metrics_df = pd.DataFrame(
            {
                "monotonicity": self.compute_monotonicity(df),
                "prognosability": self.compute_prognosability(df),
                "trendability": self.compute_trendability(df),
            }
        )
        # Use as index an incremental integer starting from 0 and add a column with the sensor names as first column
        metrics_df.reset_index(drop=True, inplace=True)
        metrics_df.insert(0, "sensor_name", self.variable_sensors)

        # Calculate the fitness as the average of the absolute values of the three metrics
        metrics_df["fitness"] = (
            metrics_df[["monotonicity", "prognosability", "trendability"]]
            .abs()
            .mean(axis=1)
        )
        return metrics_df.sort_values(by="fitness", ascending=False).reset_index(
            drop=True
        )


class CMAPSSDataSource:
    """Grain DataSource for CMAPSS data. At init time, the data is scaled and time-
    windowed across each units (engine IDs). The time windows of sensor readings are
    stored in the ``X`` attribute, whereas the ``y`` attribute contains the RUL labels
    at the end of each time window.

    Note about the scaler.

    In case the data source serves a training set, the scaler should be a fresh one. For
    test sets, the scaler should have already been fitted on the training data, so that
    the same scaling is applied to both train and test sets.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        train_or_test: str,
        scaler: StandardScaler | MinMaxScaler | None = None,
        window_size: int | None = None,
    ) -> None:
        self.df: pd.DataFrame = df
        self.scaler: StandardScaler | MinMaxScaler | None = scaler
        self.sensor_names: list[str] = [
            col for col in df.columns if col.startswith("sensor_")
        ]
        self.train_or_test: str = train_or_test
        self.unit_ids: np.ndarray = df["unit_id"].unique()
        self.window_size: int | None = window_size
        self.X: jax.Array | None = None
        self.y: jax.Array | None = None

        # input validation (train/test flag)
        if self.train_or_test not in ["train", "test"]:
            raise ValueError(
                f"train_or_test must be 'train' or 'test', got {self.train_or_test}"
            )

        if self.window_size is None:
            print(
                "Window size not specified. X and y will contain the full sequences for every unit and have"
                "the dimension relative to the time windows set to 1."
            )

        # Steps
        self._scale_sensor_data()
        self._make_data_arrays()

    def _scale_sensor_data(self) -> None:
        """Scales the sensor data in the dataframe using the provided scaler."""
        if self.scaler is None:
            return

        if self.train_or_test == "train":
            self.df[self.sensor_names] = self.scaler.fit_transform(
                self.df[self.sensor_names]
            )
        else:
            self.df[self.sensor_names] = self.scaler.transform(
                self.df[self.sensor_names]
            )

    def _make_data_arrays(self) -> None:
        """Extracts sliding time windows and corresponding RUL targets for all units in
        the dataframe.

        Stride is equal to 1, and windows are not allowed to cross unit_id boundaries.
        Notice that the RUL target for a window is the RUL at the end of that window.
        """
        unit_features = []
        unit_labels = []

        for unit_id in self.unit_ids:
            features, labels = self.get_unit_arrays(unit_id, self.window_size)
            unit_features.append(features)
            unit_labels.append(labels)

        self.X = jnp.concat(unit_features, axis=0)
        self.y = jnp.concat(unit_labels, axis=0)

    def get_unit_arrays(
        self, unit_id: int, window_size: int | None = None
    ) -> tuple[jax.Array, jax.Array]:
        """Extracts time windows and corresponding RUL targets for a specific unit.

        The window size overrides the one provided at init time.
        """
        unit_df = self.df[self.df["unit_id"] == unit_id].sort_values("time_cycles")
        features = unit_df[self.sensor_names].values
        labels = unit_df["RUL"].values

        if window_size is None:
            # If window_size is not specified, use the full sequence as a single window
            # Notice that the time-window dimension is set to 1.
            return jnp.array(features).reshape(1, *features.shape), jnp.array(
                labels
            ).reshape(1, -1)
        else:
            sequences = []
            targets = []
            for i in range(len(unit_df) - window_size + 1):
                window = features[i : i + window_size]
                window_target = labels[i + window_size - 1]
                sequences.append(window)
                targets.append(window_target)
            return jnp.array(sequences), jnp.array(targets).reshape(-1, 1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, jax.Array]:
        return self.X[record_key], self.y[record_key]
