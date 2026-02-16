from __future__ import annotations

from pathlib import Path
from typing import Generator, SupportsIndex

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


def _generate_discharge_histories(
    n_hists: int, shared_sim_config: dict
) -> Generator[tuple[np.ndarray, float]]:
    """Generate discharge histories with varying uniform random initial SoCs.

    Args:
        n_hists (int): number of discharge histories to generate.
        shared_sim_config (dict): shared simulator configuration.
            Number of simulations is set to 1.

    Yields:
        tuple[np.ndarray, float]: a tuple of (discharge_voltage_per_sim, t_eod) for
            each generated discharge history.
    """
    for _ in range(n_hists):
        sim_config = shared_sim_config.copy()
        sim_config["SoC_0"] = np.random.uniform(
            0.2, 1.0
        )  # Random initial SoC between 20% and 100%
        sim = les.SimulatorSimple(sim_config)
        sim.simulate()
        discharge_voltage_per_sim: np.ndarray = sim.v_memo.T
        yield discharge_voltage_per_sim[0], sim.t_eods[0]


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
        self, shared_sim_config: dict, n_hists: int, window_size: int, stride: int = 1
    ) -> None:
        """Data source that provides time-windowed, labelled chunks of the discharge
        history. The features are the voltage values in the time window, and the target
        is the RUL at the time step immediately following the time window.

        Args:
            shared_sim_config (dict): shared simulator configuration dict. Number of
                simulations is set to 1.
            n_hists (int): number of discharge histories to generate.
            window_size (int): the size of the time window (number of time steps).
            stride (int): the stride of the sliding time window
                (number of time steps to move the window at each step).
                Defaults to 1.

        Raises:
            ValueError: if window_size is greater than the number of time steps.
        """
        self.X = []
        self.y = []
        for discharge_voltage, t_eod in _generate_discharge_histories(
            n_hists=n_hists, shared_sim_config=shared_sim_config
        ):
            N_t = discharge_voltage.shape[0]
            # calculate RULs (labels) for this time history, using a linear degradation model.
            ruls = _back_calculate_rul_linear(t_eod=t_eod, N_t=N_t)
            # if window_size > N_t, return a single window with all the hisory and RUL=0.
            if window_size > N_t:
                self.X.append(discharge_voltage.reshape(1, -1))
                self.y.append(0.0)
                continue
            # Generate windows and targets for this discharge history.
            for start in range(0, N_t - window_size, stride):
                end = start + window_size
                self.X.append(discharge_voltage[start:end].reshape(1, -1))
                self.y.append(float(ruls[end]) if end < N_t else 0.0)

        self.X = jnp.array(self.X)
        self.y_max = jnp.max(jnp.array(self.y))
        self.y = jnp.array(self.y) / self.y_max  # Normalize RUL values to [0, 1]

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
