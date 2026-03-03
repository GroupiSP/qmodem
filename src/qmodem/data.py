from __future__ import annotations

from pathlib import Path
from typing import SupportsIndex

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import numpy as np

BATT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "battery_config.json"


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
        """Data source that provides time-windowed, labelled chunks of discharge
        histories. The simulator is called once with ``t_0=0.0``. The features are the
        voltage values in the time window, and the target is the RUL at the time step
        immediately following the time window.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator
                from lib_eod_simulation. It must be fully configured (including
                ``N_simu``) before being passed to this class.
            window_size (int): the size of the time window (number of time steps).
            stride (int): the stride of the sliding time window
                (number of time steps to move the window at each step).
                Defaults to 1.
            normalize (bool): normalizes the RUL values (divide by max(RUL)).
                Defaults to False.

        Note:
            Discharge histories shorter than ``window_size`` are skipped.
        """
        simulator.simulate(t_0=0.0)

        # Transpose for convenience. Shape=(N_simu, N_t).
        discharge_voltage_per_sim: np.ndarray = simulator.v_memo.T
        N_t = discharge_voltage_per_sim.shape[1]

        self.X = []
        self.y = []
        self.n_skipped = 0
        for i in range(simulator.N_simu):
            discharge_voltage = discharge_voltage_per_sim[i]
            t_eod = simulator.t_eods[i]
            # Calculate RULs (labels) using a linear degradation model (t_0=0.0).
            ruls = _back_calculate_rul_linear(t_eod=t_eod, N_t=N_t)
            # Skip histories shorter than the window size.
            if window_size > N_t:
                self.n_skipped += 1
                continue
            # Generate windows and targets for this discharge history.
            for start in range(0, N_t - window_size, stride):
                end = start + window_size
                self.X.append(discharge_voltage[start:end].reshape(1, -1))
                self.y.append(float(ruls[end]) if end < N_t else 0.0)

            # Handle the last window if it doesn't fit perfectly.
            if end < N_t:
                self.X.append(discharge_voltage[-window_size:].reshape(1, -1))
                self.y.append(0.0)

        self.X = jnp.array(self.X)
        y_array = jnp.array(self.y)
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
        return self.X[record_key], self.y[record_key].reshape(-1)
