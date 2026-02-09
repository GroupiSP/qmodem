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
    ruls[-1] = 0.0  # Ensure last RUL is exactly 0.
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
    ) -> None:
        """Data source that provides time-windowed, labelled chunks of the discharge
        history. The features are the voltage values in the time window, and the target
        is the RUL at the time step immediately following the time window (sliding time
        window strategy). The last time window is returned with a RUL of 0.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source.
            window_size (int): the size of the time window (number of time steps).
            stride (int): the stride of the sliding time window
                (number of time steps to move the window at each step).
                Defaults to 1.

        Raises:
            InputError: if the simulator's number of simulations is greater than 1.
        """
        if simulator.N_simu > 1:
            raise ValueError(
                "BatterySimulationTimeWindowSource only supports a single simulation. "
            )

        simulator.simulate()
        # Shape=(N_simu, N_t).
        discharge_voltage_per_sim: np.ndarray = simulator.v_memo.T
        N_t = discharge_voltage_per_sim.shape[1]
        # Only one simulation, so we can take the first row.
        self.discharge_voltage = jnp.array(discharge_voltage_per_sim[0])
        self.ruls = _back_calculate_rul_linear(t_eod=simulator.t_eods[0], N_t=N_t)
        self.window_size = window_size
        self.stride = stride
        self.N_t = N_t

    def __iter__(self):
        """Returns an iterator over the records in the dataset."""
        for start in range(0, self.N_t - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = self.discharge_voltage[start:end].reshape(1, -1)
            if end < self.N_t:
                target = self.ruls[end]
            else:
                # Last window, return with RUL of 0.
                target = jnp.array(0.0)
            yield (window, target)


# TODO: remove
class BatterySimulationTimeSeriesSource:
    def __init__(
        self,
        simulator: les.SimulatorSimple | les.SimulatorComplete,
    ) -> None:
        """Runs and access to battery simulation data as time series.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source.
        """
        simulator.simulate()

        # Transpose for convenience. Shape=(N_simu, N_t).
        discharge_voltage_per_sim: np.ndarray = simulator.v_memo.T

        self.X = jnp.array(discharge_voltage_per_sim)
        self.y = jnp.array(simulator.t_eods)

    def __len__(self) -> int:
        """Number of records in the dataset, corresponding to the number of discharge
        histories."""
        return len(self.y)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, float]:
        """Retrieves record for the given record_key."""
        return self.X[record_key], self.y[record_key]

    def shuffle(self, seed: int) -> None:
        """Shuffles the order of the records in the dataset."""
        key = jax.random.PRNGKey(seed)
        perm = jax.random.permutation(key, len(self.y))
        self.X = self.X[perm]
        self.y = self.y[perm]
