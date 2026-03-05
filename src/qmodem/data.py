from __future__ import annotations

from pathlib import Path
from typing import Any, SupportsIndex

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
