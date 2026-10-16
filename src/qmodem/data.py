from __future__ import annotations

import typing
from collections.abc import Callable, Sequence
from enum import StrEnum, auto
from typing import Any, Protocol, SupportsIndex

import jax
import jax.numpy as jnp
import jaxtyping
import numpy as np
import pandas as pd


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
        return features.reshape(1, window_size, -1), [0.0]

    windows: list[np.ndarray] = []
    targets: list[float] = []

    # NOTE The second to last and last window may overlap
    start = 0
    end = start + window_size
    while end < N_t:
        windows.append(features[start:end])
        targets.append(float(ruls[end]))

        start += stride
        end = start + window_size

    # right pad the last window and assign a RUL of 0.0
    last_window = features[start:]
    if last_window.shape[0] < window_size:
        pad_len = window_size - last_window.shape[0]
        last_window = np.concatenate(
            [
                last_window,
                np.full(shape=(pad_len, features.shape[1]), fill_value=last_window[-1]),
            ]
        )
    windows.append(last_window)
    targets.append(0.0)

    return windows, targets


class DataSource(Protocol):
    """Protocol for indexable sources of ``(features, targets)`` records."""

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
    def fit(self, x: pd.DataFrame, y: Any) -> None: ...
    def fit_transform(self, x: pd.DataFrame, y: Any) -> pd.DataFrame: ...
    def transform(self, x: pd.DataFrame) -> pd.DataFrame: ...
    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame: ...


class IdentityScaler:
    """A scaler that performs no scaling, returning the input as-is."""

    def fit(self, x: pd.DataFrame, y: Any) -> None:
        """No-op for fitting the scaler."""

    def fit_transform(self, x: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        """Returns the input arrays as-is without any scaling."""
        return x

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Returns the input array as-is without any scaling."""
        return x

    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
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
    def __init__(
        self,
        scaler: DataScaler | None = None,
        features: str | Sequence[str] | None = None,
    ) -> None:
        """A data processing step that scales features and targets using provided
        scalers. Meant to be used in a data pipeline.

        Args:
            scaler: A scaler object that implements the DataScaler protocol. If None, an IdentityScaler is used.
            features: A string or list of strings specifying the feature names to be scaled. If None, scaling is applied
                to all features of the dataset
        """
        self.scaler = scaler if scaler is not None else IdentityScaler()
        self.features = features
        self._mode = ScalingMode.FIT_TRANSFORM

    @property
    def mode(self) -> ScalingMode:
        return self._mode

    @mode.setter
    def mode(self, value: ScalingMode) -> None:
        if not isinstance(value, ScalingMode):
            raise TypeError(f"mode must be an instance of ScalingMode, got {value}")
        self._mode = value

    def _fit_transform(self, x: pd.DataFrame, y: Any = None) -> pd.DataFrame:
        features = self.features if self.features is not None else x.columns
        x = x.copy()
        x[features] = self.scaler.fit_transform(x[features], y)
        return x

    def _transform(self, x: pd.DataFrame) -> pd.DataFrame:
        features = self.features if self.features is not None else x.columns
        x = x.copy()
        x[features] = self.scaler.transform(x[features])
        return x

    _dispatch: typing.ClassVar = {
        ScalingMode.FIT_TRANSFORM: _fit_transform,
        ScalingMode.TRANSFORM: _transform,
    }

    def __call__(self, x: pd.DataFrame) -> pd.DataFrame:
        return self._dispatch[self._mode](self, x)


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
    feature_windows: list[np.ndarray] = []
    rul_targets: list[float] = []

    unit_ids = df["run_id"].unique()
    for unit_id in unit_ids:
        unit_df = df[df["run_id"] == unit_id].sort_values("time")
        feature_array = unit_df[features].values
        ruls = unit_df["rul"].values

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
