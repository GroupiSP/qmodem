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
        n_rul_samples: int = 1,
        rul_sim_config: dict | None = None,
    ) -> None:
        """Data source that provides time-windowed, labelled chunks of the discharge
        history. The features are the voltage values in the time window, and the target
        is the RUL at the time step immediately following the time window.

        When ``n_rul_samples > 1``, RUL targets are obtained by running stochastic
        forward simulations from the SoC at each window endpoint. Each voltage window
        is replicated ``n_rul_samples`` times, each paired with a different sampled
        RUL target. This produces position-dependent (heteroscedastic) target variance:
        windows at high SoC have large RUL spread, windows near end-of-discharge have
        small spread.

        When ``n_rul_samples == 1`` (default), a single forward simulation is run per
        window position and ``rul_sim_config`` must still be provided, unless the
        caller wants the legacy linear back-calculation (pass ``rul_sim_config=None``).

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
            n_rul_samples (int): number of stochastic forward simulations to run per
                window position for RUL target sampling. Each window is replicated
                this many times. Defaults to 1.
            rul_sim_config (dict | None): simulator configuration dict used to run
                forward RUL simulations. Must contain keys: ``battery``,
                ``discharge_policy``, ``v_cut``, ``dt``, ``omega_std``, ``eta_std``.
                Required when ``n_rul_samples >= 1`` and forward simulation is desired.
                When None, falls back to linear RUL back-calculation (legacy behavior).

        Raises:
            ValueError: if the simulator's number of simulations is greater than 1.
            ValueError: if window_size is greater than the number of time steps.
            ValueError: if n_rul_samples > 1 but rul_sim_config is None.
        """
        if simulator.N_simu > 1:
            raise ValueError(
                "BatterySimulationTimeWindowSource only supports a single simulation."
            )
        if n_rul_samples > 1 and rul_sim_config is None:
            raise ValueError("rul_sim_config is required when n_rul_samples > 1.")

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

        # Determine window positions
        num_windows = (N_t - window_size) // stride + 1
        window_positions: list[tuple[int, int]] = []
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            window_positions.append((start, end))

        # Add a final backwards-extended window that reaches the true end of
        # the voltage trace, so the model sees near-EoD voltage patterns.
        last_regular_end = (num_windows - 1) * stride + window_size
        if last_regular_end < N_t:
            window_positions.append((N_t - window_size, N_t))

        # Compute RUL targets
        if rul_sim_config is not None:
            soc_trajectory: np.ndarray = simulator.soc_memo.T[0]
            targets_per_window = self._simulate_rul_targets(
                window_positions, N_t, soc_trajectory, n_rul_samples, rul_sim_config
            )
        else:
            ruls = _back_calculate_rul_linear(t_eod=simulator.t_eods[0], N_t=N_t)
            targets_per_window = []
            for _start, end in window_positions:
                target = float(ruls[end]) if end < N_t else 0.0
                targets_per_window.append([target])

        # Build replicated windows and targets
        windows = []
        targets = []
        for (start, end), rul_samples in zip(window_positions, targets_per_window):
            window = discharge_voltage[start:end].reshape(1, -1)
            for rul in rul_samples:
                windows.append(window)
                targets.append(rul)

        self.n_rul_samples = n_rul_samples if rul_sim_config is not None else 1
        self.X = jnp.array(np.array(windows))
        self.y = jnp.array(np.array(targets))
        self.y_max = jnp.max(self.y)

        if normalize:
            self.y = self.y / self.y_max

    @staticmethod
    def _simulate_rul_targets(
        window_positions: list[tuple[int, int]],
        n_t: int,
        soc_trajectory: np.ndarray,
        n_rul_samples: int,
        rul_sim_config: dict,
    ) -> list[list[float]]:
        """Run stochastic forward simulations to obtain RUL targets per window.

        Args:
            window_positions: list of (start, end) index pairs for each window.
            n_t: total number of time steps in the primary simulation.
            soc_trajectory: SoC values at each time step from the primary simulation.
            n_rul_samples: number of forward simulations per window position.
            rul_sim_config: configuration dict with keys: battery, discharge_policy,
                v_cut, dt, omega_std, eta_std.

        Returns:
            List of lists, where each inner list contains ``n_rul_samples`` sampled
            RUL values for that window position.
        """
        targets_per_window: list[list[float]] = []
        for _start, end in window_positions:
            if end >= n_t:
                # Final window at end-of-discharge: RUL is 0
                targets_per_window.append([0.0] * n_rul_samples)
                continue

            soc_at_end = float(soc_trajectory[end])
            sim_config = {
                "N_simu": n_rul_samples,
                "v_cut": rul_sim_config["v_cut"],
                "SoC_0": soc_at_end,
                "dt": rul_sim_config["dt"],
                "omega_std": rul_sim_config["omega_std"],
                "eta_std": rul_sim_config["eta_std"],
                "I": rul_sim_config["discharge_policy"],
                "battery": rul_sim_config["battery"],
            }
            fwd_sim = les.SimulatorSimple(sim_config)
            fwd_sim.simulate()
            targets_per_window.append(list(fwd_sim.t_eods))

        return targets_per_window

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
