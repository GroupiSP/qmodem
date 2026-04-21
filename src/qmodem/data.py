from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, SupportsIndex

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import numpy as np
import pandas as pd
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import CMAPSS_DIR_PATH


def _back_calculate_rul_linear(t_eod: float, N_t: int, t_0: float = 0.0) -> np.ndarray:
    """Back-calculates RUL values for a linear degradation model.

    Args:
        t_eod (float): time of end of discharge (failure).
        N_t (int): number of time steps in the discharge history.
        t_0 (float): initial time of the discharge history. Defaults to 0.0.

    Returns:
        np.ndarray: RUL values for each time step, clipped to be non-negative.
    """
    ruls = np.linspace(t_eod - t_0, 0.0, N_t)
    return jnp.array(ruls)


def _run_discharge(config: dict[str, Any], soc_0: float) -> tuple[np.ndarray, float]:
    """Run a single discharge simulation.

    Args:
        config: Simulator configuration dictionary (any ``N_simu`` / ``SoC_0``
            values are overridden).
        soc_0: Initial state of charge for this discharge.

    Returns:
        A tuple of ``(voltage_history, t_eod)`` where *voltage_history* is a
        1-D array of shape ``(N_t,)`` and *t_eod* is the end-of-discharge time.
    """
    sim_config = config.copy()
    sim_config["N_simu"] = 1
    sim_config["SoC_0"] = soc_0
    sim = les.SimulatorSimple(sim_config)
    sim.simulate()
    return sim.v_memo.flatten(), float(sim.t_eods[0])


def _make_windows(
    voltage: np.ndarray,
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
        voltage: 1-D voltage history of shape ``(N_t,)``.
        ruls: 1-D RUL values of shape ``(N_t,)`` aligned with *voltage*.
        window_size: Number of time steps per window.
        stride: Step size for the sliding window.

    Returns:
        A tuple ``(windows, targets)`` where each window has shape
        ``(1, window_size)`` and each target is a scalar RUL value.
    """
    N_t = len(voltage)

    # Left-edge-pad short histories so at least one full window can be made.
    if N_t < window_size:
        pad_len = window_size - N_t
        voltage = np.concatenate([np.full(pad_len, voltage[0]), voltage])
        ruls = np.concatenate([np.full(pad_len, ruls[0]), ruls])
        N_t = window_size

    windows: list[np.ndarray] = []
    targets: list[float] = []

    end = 0
    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        windows.append(voltage[start:end].reshape(1, -1))
        targets.append(float(ruls[end]) if end < N_t else 0.0)

    # Trailing window covering the very end of the history.
    if end < N_t:
        windows.append(voltage[-window_size:].reshape(1, -1))
        targets.append(0.0)

    return windows, targets


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


def create_dataloaders(
    ds_train: DataSource,
    ds_val: DataSource,
    batch_size: int,
    seed_train: int,
    seed_val: int,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    *,
    drop_remainder: bool = False,
) -> tuple[Any, Any]:
    """Create Grain DataLoaders for training and validation."""

    sampler_train = IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=shuffle_train, seed=seed_train
    )
    dataloader_train = DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
        worker_count=0,
    )

    sampler_val = IndexSampler(
        num_records=len(ds_val), num_epochs=1, shuffle=shuffle_val, seed=seed_val
    )
    dataloader_val = DataLoader(
        data_source=ds_val,
        sampler=sampler_val,
        operations=[Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
        worker_count=0,
    )

    return dataloader_train, dataloader_val


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


class BatterySimulationTimeWindowSource:
    def __init__(
        self,
        simulator_config: dict[str, Any],
        n_histories: int,
        window_size: int,
        stride: int = 1,
        normalize: bool = False,
        soc_range: tuple[float, float] = (0.05, 1.0),
    ) -> None:
        """Data source that provides time-windowed, labelled chunks of discharge
        histories. Each history starts from a ``SoC_0`` sampled uniformly from
        *soc_range*. A separate simulation with ``N_simu=1`` is run for every history.

        The features are the voltage values in the time window, and the target
        is the RUL at the time step immediately following the window.

        Args:
            simulator_config: Base simulator configuration dictionary.  The
                ``N_simu`` and ``SoC_0`` entries are overridden internally.
            n_histories: Number of independent discharge histories to generate.
            window_size: The size of the time window (number of time steps).
            stride: The stride of the sliding time window (number of time steps
                to move the window at each step). Defaults to 1.
            normalize: Normalizes the RUL values (divide by max(RUL)).
                Defaults to False.
            soc_range: ``(low, high)`` bounds for the uniform SoC₀ sampling.
                Defaults to ``(0.05, 1.0)``.

        Note:
            Discharge histories shorter than *window_size* are left-edge-padded
            (first voltage value repeated) so they still contribute at least one
            window.
        """
        all_windows: list[np.ndarray] = []
        all_targets: list[float] = []
        self.soc_0s: list[float] = []

        for _ in range(n_histories):
            soc_0 = float(np.random.uniform(*soc_range))
            self.soc_0s.append(soc_0)

            voltage, t_eod = _run_discharge(simulator_config, soc_0)
            ruls = _back_calculate_rul_linear(t_eod=t_eod, N_t=len(voltage))

            windows, targets = _make_windows(voltage, ruls, window_size, stride)
            all_windows.extend(windows)
            all_targets.extend(targets)

        self.X = jnp.array(all_windows)
        y_array = jnp.array(all_targets)
        self.y_max = jnp.max(y_array)

        if normalize:
            self.y = y_array / self.y_max
        else:
            self.y = y_array

    def __len__(self) -> int:
        """Number of time windows in the dataset."""
        return len(self.y)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, jax.Array]:
        """Retrieves window and target for the given record_key.

        Args:
            record_key (SupportsIndex): Index of the window to retrieve.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple of (window, target) where window
                has shape (1, window_size) and target has shape (1,) for a scalar
                index or (batch_size,) for a slice.
        """
        return self.X[record_key], self.y[record_key]

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        window_size: int,
        stride: int = 1,
        normalize: bool = False,
    ) -> BatterySimulationTimeWindowSource:
        """Create a data source from a pre-generated ``.npz`` file.

        The file must contain ``voltages`` (object array of 1-D voltage
        histories) and ``t_eods`` (1-D array of end-of-discharge times),
        as produced by :func:`qmodem.generate.generate_train_data`.

        Args:
            path: Path to the ``.npz`` file.
            window_size: The size of the time window (number of time steps).
            stride: The stride of the sliding time window. Defaults to 1.
            normalize: Normalizes the RUL values (divide by max(RUL)).
                Defaults to False.

        Returns:
            A populated ``BatterySimulationTimeWindowSource`` instance.
        """
        data = np.load(path, allow_pickle=True)
        voltages = data["voltages"]
        t_eods = data["t_eods"]
        soc_0s = data["soc_0s"] if "soc_0s" in data else []

        obj = cls.__new__(cls)
        all_windows: list[np.ndarray] = []
        all_targets: list[float] = []
        obj.soc_0s = list(soc_0s)

        for voltage, t_eod in zip(voltages, t_eods):
            ruls = _back_calculate_rul_linear(t_eod=float(t_eod), N_t=len(voltage))
            windows, targets = _make_windows(voltage, ruls, window_size, stride)
            all_windows.extend(windows)
            all_targets.extend(targets)

        obj.X = jnp.array(all_windows)
        y_array = jnp.array(all_targets)
        obj.y_max = jnp.max(y_array)

        if normalize:
            obj.y = y_array / obj.y_max
        else:
            obj.y = y_array

        return obj


class CMAPSSAnalyst:
    """Loads, preprocesses and analyses the CMAPSS FD001/train dataset.

    Attributes:
        df: The full dataframe loaded from the original CMAPSS FD001/train file, with the RUL column added.
        variable_sensors: The list of sensor column names that are not constant across the whole dataset.
    """

    constant_sensors: list[str] = [f"sensor_{i}" for i in [1, 5, 6, 10, 16, 18, 19]]
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

    def __init__(self) -> None:
        # Define the attributes
        self.df: pd.DataFrame | None = None
        self.variable_sensors: list[str] = [
            f"sensor_{i}"
            for i in range(1, 22)
            if f"sensor_{i}" not in self.constant_sensors
        ]

        # Steps
        self._load_cmapss_fd001_train()
        self._add_rul()
        self._exclude_constant_sensors()

    def _load_cmapss_fd001_train(
        self,
    ) -> None:
        # load the data
        self.df = pd.read_csv(
            CMAPSS_DIR_PATH / "train_FD001.txt",
            sep=r"\s+",
            header=None,
            names=self.column_names,
        )

    def _add_rul(self) -> None:
        # add the RUL column
        self.df["RUL"] = self.df.groupby("unit_id")["time_cycles"].transform(
            lambda x: x.max() - x
        )

    def _exclude_constant_sensors(self) -> None:
        # drop the constant sensors
        self.df.drop(columns=self.constant_sensors, inplace=True)

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
