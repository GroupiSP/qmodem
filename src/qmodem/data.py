from __future__ import annotations

from pathlib import Path
from typing import SupportsIndex

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import numpy as np

BATT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "battery_config.json"


def _back_calculate_rul_linear(t_eod: float, N_t: int) -> np.ndarray:
    """Back-calculates RUL values for a linear degradation model.

    Args:
        t_eod (float): time of end of discharge (failure).
        N_t (int): number of time steps in the discharge history.

    Returns:
        np.ndarray: RUL values for each time step, clipped to be non-negative.
    """
    dt = t_eod / N_t
    ruls = np.clip(t_eod - np.arange(N_t) * dt, a_min=0.0, a_max=None)
    return jnp.array(ruls)


class BatterySimulationSource:
    def __init__(
        self,
        simulator: les.SimulatorSimple | les.SimulatorComplete,
        normalize: bool = False,
    ) -> None:
        """Runs and access to battery simulation data. The histories of multiple
        simulations are flattened into a single dataset, where each record corresponds
        to a single time step in a discharge history. The features are the voltage at
        that time step, and the target is the RUL at that time step.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source.
            normalize (bool): Normalizes the RUL values (divide by max(RUL)). Defaults to False.
        """
        simulator.simulate()

        # Transpose for convenience. Shape=(N_simu, N_t).
        discharge_voltage_per_sim: np.ndarray = simulator.v_memo.T
        N_t = discharge_voltage_per_sim.shape[1]

        X = discharge_voltage_per_sim.flatten().reshape(-1, 1)
        ruls = np.empty(shape=(simulator.N_simu * N_t))

        for i in range(simulator.N_simu):
            ruls[i * N_t : (i + 1) * N_t] = _back_calculate_rul_linear(
                t_eod=simulator.t_eods[i], N_t=N_t
            )
        y = ruls
        self.y_max = np.max(y)

        if normalize:
            y /= self.y_max

        self.X = jnp.array(X)
        self.y = jnp.array(y)

    def __len__(self) -> int:
        """Number of records in the dataset."""
        return len(self.y)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, float]:
        """Retrieves record for the given record_key."""
        return self.X[record_key], self.y[record_key]


class BatterySimulationTimeWindowSource:
    def __init__(
        self,
        simulator: les.SimulatorSimple | les.SimulatorComplete,
        window_size: int,
        stride: int = 1,
        normalize: bool = False,
    ) -> None:
        """Data source that provides time-windowed, labelled chunks of the discharge
        history. The features are the voltage values in the time window, and the target
        is the RUL at the time step immediately following the time window (sliding time
        window strategy). The last time window is returned with a RUL of 0.

        This data source is compatible with Google Grain's DataLoader via random access
        through __getitem__ and __len__ methods.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source.
            window_size (int): the size of the time window (number of time steps).
            stride (int): the stride of the sliding time window
                (number of time steps to move the window at each step).
                Defaults to 1.
            normalize (bool): Normalizes the RUL values (divide by max(RUL)).
                Defaults to False.

        Raises:
            ValueError: if the simulator's number of simulations is greater than 1.
            ValueError: if window_size is greater than the number of time steps.
        """
        if simulator.N_simu > 1:
            raise ValueError(
                "BatterySimulationTimeWindowSource only supports a single simulation."
            )

        simulator.simulate()
        # Shape=(N_simu, N_t).
        discharge_voltage_per_sim: np.ndarray = simulator.v_memo.T
        N_t = discharge_voltage_per_sim.shape[1]

        if window_size > N_t:
            raise ValueError(
                f"window_size ({window_size}) cannot be greater than "
                f"number of time steps ({N_t})."
            )

        # Only one simulation, so we can take the first row.
        discharge_voltage = discharge_voltage_per_sim[0]
        ruls = _back_calculate_rul_linear(t_eod=simulator.t_eods[0], N_t=N_t)

        # Pre-compute all windows and targets
        num_windows = (N_t - window_size) // stride + 1
        windows = []
        targets = []

        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            window = discharge_voltage[start:end].reshape(1, -1)
            windows.append(window)

            # Target is RUL at the next time step after window
            if end < N_t:
                target = ruls[end]
            else:
                target = 0.0
            targets.append(target)

        # Add a final backwards-extended window that reaches the true end of
        # the voltage trace, so the model sees near-EoD voltage patterns.
        last_regular_end = (num_windows - 1) * stride + window_size
        if last_regular_end < N_t:
            start = N_t - window_size
            window = discharge_voltage[start:N_t].reshape(1, -1)
            windows.append(window)
            targets.append(0.0)

        self.X = jnp.array(np.array(windows))
        self.y = jnp.array(np.array(targets))
        self.y_max = jnp.max(self.y)

        if normalize:
            self.y = self.y / self.y_max

    def __len__(self) -> int:
        """Number of time windows in the dataset."""
        return len(self.y)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, jax.Array]:
        """Retrieves window and target for the given record_key.

        Args:
            record_key (SupportsIndex): Index of the window to retrieve.

        Returns:
            tuple[jax.Array, jax.Array]: A tuple of (window, target) where window
                has shape (1, window_size) and target is a scalar.
        """
        return self.X[record_key], self.y[record_key]


class CombinedTimeWindowSource:
    def __init__(self, sources: list[BatterySimulationTimeWindowSource]) -> None:
        """Combined data source from multiple time window sources.

        Combines multiple BatterySimulationTimeWindowSource instances into one dataset
        by concatenating all windows and targets. This allows training on multiple
        discharge histories.

        Args:
            sources (list[BatterySimulationTimeWindowSource]): List of time window
                data sources to combine.
        """
        all_windows = []
        all_targets = []

        for source in sources:
            all_windows.append(source.X)
            all_targets.append(source.y)

        self.X = jnp.concatenate(all_windows, axis=0)
        self.y = jnp.concatenate(all_targets, axis=0)

    def __len__(self) -> int:
        """Number of records in the combined dataset."""
        return len(self.y)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, jax.Array]:
        """Retrieves window and target for the given record_key."""
        return self.X[record_key], self.y[record_key]
