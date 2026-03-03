"""Tests for qmodem.data module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from qmodem.data import BatterySimulationTimeWindowSource


def _make_mock_simulator(n_simu: int, n_t: int, t_eod: float = 5.0) -> MagicMock:
    """Build a mock simulator with ``n_simu`` discharge histories of length ``n_t``."""
    simulator = MagicMock()
    simulator.N_simu = n_simu
    # v_memo has shape (N_t, N_simu) — same convention as lib_eod_simulation
    simulator.v_memo = np.linspace(4.2, 3.0, n_t * n_simu).reshape(n_t, n_simu)
    simulator.t_eods = np.full(n_simu, t_eod)
    return simulator


def _windows_per_history(n_t: int, window_size: int, stride: int) -> int:
    """Return the number of windows produced from a single history of length ``n_t``."""
    count = 0
    end = 0
    for start in range(0, n_t - window_size, stride):
        end = start + window_size
        count += 1
    if count > 0 and end < n_t:
        count += 1  # trailing window
    return count


@pytest.mark.parametrize("n_simu", [1, 3, 5])
def test_time_window_source_uses_all_simulations(n_simu: int) -> None:
    """BatterySimulationTimeWindowSource must produce windows from every simulation."""
    n_t = 10
    window_size = 3
    stride = 1
    simulator = _make_mock_simulator(n_simu=n_simu, n_t=n_t)

    source = BatterySimulationTimeWindowSource(
        simulator, window_size=window_size, stride=stride
    )

    expected_windows = n_simu * _windows_per_history(n_t, window_size, stride)
    assert len(source) == expected_windows, (
        f"Expected {expected_windows} windows for {n_simu} simulation(s), "
        f"got {len(source)}"
    )


def test_time_window_source_normalize_true() -> None:
    """With normalize=True all targets must be in [0, 1]."""
    simulator = _make_mock_simulator(n_simu=2, n_t=10)
    source = BatterySimulationTimeWindowSource(simulator, window_size=3, normalize=True)
    assert float(np.max(source.y)) <= 1.0
    assert float(np.min(source.y)) >= 0.0


def test_time_window_source_normalize_false() -> None:
    """With normalize=False targets are raw RUL values (max > 1 for realistic t_eod)."""
    simulator = _make_mock_simulator(n_simu=2, n_t=10, t_eod=5.0)
    source = BatterySimulationTimeWindowSource(
        simulator, window_size=3, normalize=False
    )
    assert float(np.max(source.y)) > 1.0


def test_time_window_source_getitem_single_shape() -> None:
    """__getitem__ with a scalar index returns window (1, window_size) and scalar
    target."""
    window_size = 4
    simulator = _make_mock_simulator(n_simu=1, n_t=12)
    source = BatterySimulationTimeWindowSource(simulator, window_size=window_size)

    window, target = source[0]
    assert window.shape == (1, window_size)
    assert target.shape == (1,)


def test_time_window_source_getitem_batch_shape() -> None:
    """__getitem__ with a slice returns windows (batch, 1, window_size) and targets
    (batch,)."""
    window_size = 4
    batch_size = 3
    simulator = _make_mock_simulator(n_simu=1, n_t=12)
    source = BatterySimulationTimeWindowSource(simulator, window_size=window_size)

    windows, targets = source[:batch_size]
    assert windows.shape == (batch_size, 1, window_size)
    assert targets.shape == (batch_size,)
