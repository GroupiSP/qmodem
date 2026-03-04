"""Tests for qmodem.data module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qmodem.data import (
    BatterySimulationTimeWindowSource,
    _back_calculate_rul_linear,
    _make_windows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_simulator(n_t: int, t_eod: float = 5.0) -> MagicMock:
    """Build a mock ``SimulatorSimple`` that returns a single discharge history of
    length *n_t*."""
    simulator = MagicMock()
    simulator.N_simu = 1
    # v_memo has shape (N_t, 1) — same convention as lib_eod_simulation
    simulator.v_memo = np.linspace(4.2, 3.0, n_t).reshape(n_t, 1)
    simulator.t_eods = np.array([t_eod])
    return simulator


def _patch_run_discharge(n_t: int, t_eod: float = 5.0):
    """Return a *patch* context-manager that replaces ``_run_discharge`` with a
    deterministic stub producing a history of length *n_t*."""
    voltage = np.linspace(4.2, 3.0, n_t)

    def _stub(_config, _soc_0):
        return voltage.copy(), t_eod

    return patch("qmodem.data._run_discharge", side_effect=_stub)


def _windows_per_history(n_t: int, window_size: int, stride: int) -> int:
    """Return the number of windows produced from a single history of length *n_t*
    (including padding when ``n_t < window_size``)."""
    # Padding extends length to at least window_size
    effective_n_t = max(n_t, window_size)
    count = 0
    end = 0
    for start in range(0, effective_n_t - window_size, stride):
        end = start + window_size
        count += 1
    if count > 0 and end < effective_n_t:
        count += 1  # trailing window
    # If no loop iterations happened, still one window (padded case or exact fit)
    if count == 0:
        count = 1
    return count


# ---------------------------------------------------------------------------
# Tests for _back_calculate_rul_linear
# ---------------------------------------------------------------------------


class TestBackCalculateRulLinear:
    def test_basic(self) -> None:
        ruls = _back_calculate_rul_linear(t_eod=10.0, N_t=5)
        assert ruls.shape == (5,)
        np.testing.assert_allclose(ruls[0], 10.0)
        np.testing.assert_allclose(ruls[-1], 0.0)

    def test_with_t0(self) -> None:
        ruls = _back_calculate_rul_linear(t_eod=10.0, N_t=3, t_0=2.0)
        np.testing.assert_allclose(ruls[0], 8.0)
        np.testing.assert_allclose(ruls[-1], 0.0)


# ---------------------------------------------------------------------------
# Tests for _make_windows
# ---------------------------------------------------------------------------


class TestMakeWindows:
    def test_basic_shape(self) -> None:
        voltage = np.linspace(4.2, 3.0, 10)
        ruls = np.linspace(5.0, 0.0, 10)
        windows, targets = _make_windows(voltage, ruls, window_size=3, stride=1)
        for w in windows:
            assert w.shape == (1, 3)
        assert len(windows) == len(targets)

    def test_window_count(self) -> None:
        """Verify the expected window count for a normal-length history."""
        n_t, ws, stride = 10, 3, 1
        voltage = np.linspace(4.2, 3.0, n_t)
        ruls = np.linspace(5.0, 0.0, n_t)
        windows, _ = _make_windows(voltage, ruls, window_size=ws, stride=stride)
        expected = _windows_per_history(n_t, ws, stride)
        assert len(windows) == expected

    def test_short_history_padded(self) -> None:
        """A history shorter than window_size must still produce at least one window via
        left-edge-padding."""
        voltage = np.array([4.0, 3.5])
        ruls = np.array([2.0, 0.0])
        windows, targets = _make_windows(voltage, ruls, window_size=5, stride=1)
        assert len(windows) >= 1
        assert windows[0].shape == (1, 5)
        # First three values should be the edge-padded initial voltage
        np.testing.assert_allclose(windows[0][0, :3], 4.0)

    def test_last_target_is_zero(self) -> None:
        """The trailing window target is always 0 (end-of-discharge)."""
        voltage = np.linspace(4.2, 3.0, 12)
        ruls = np.linspace(5.0, 0.0, 12)
        _, targets = _make_windows(voltage, ruls, window_size=4, stride=3)
        assert targets[-1] == 0.0

    def test_stride_reduces_windows(self) -> None:
        n_t = 20
        voltage = np.linspace(4.2, 3.0, n_t)
        ruls = np.linspace(5.0, 0.0, n_t)
        w1, _ = _make_windows(voltage, ruls, window_size=4, stride=1)
        w2, _ = _make_windows(voltage, ruls, window_size=4, stride=5)
        assert len(w2) < len(w1)


# ---------------------------------------------------------------------------
# Tests for BatterySimulationTimeWindowSource
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_histories", [1, 3, 5])
def test_time_window_source_uses_all_histories(n_histories: int) -> None:
    """BatterySimulationTimeWindowSource must produce windows from every history."""
    n_t = 10
    window_size = 3
    stride = 1

    with _patch_run_discharge(n_t):
        source = BatterySimulationTimeWindowSource(
            simulator_config={},
            n_histories=n_histories,
            window_size=window_size,
            stride=stride,
        )

    expected_windows = n_histories * _windows_per_history(n_t, window_size, stride)
    assert len(source) == expected_windows, (
        f"Expected {expected_windows} windows for {n_histories} history(ies), "
        f"got {len(source)}"
    )


def test_time_window_source_normalize_true() -> None:
    """With normalize=True all targets must be in [0, 1]."""
    with _patch_run_discharge(n_t=10):
        source = BatterySimulationTimeWindowSource(
            simulator_config={},
            n_histories=2,
            window_size=3,
            normalize=True,
        )
    assert float(np.max(source.y)) <= 1.0
    assert float(np.min(source.y)) >= 0.0


def test_time_window_source_normalize_false() -> None:
    """With normalize=False targets are raw RUL values (max > 1 for realistic t_eod)."""
    with _patch_run_discharge(n_t=10, t_eod=5.0):
        source = BatterySimulationTimeWindowSource(
            simulator_config={},
            n_histories=2,
            window_size=3,
            normalize=False,
        )
    assert float(np.max(source.y)) > 1.0


def test_time_window_source_getitem_single_shape() -> None:
    """__getitem__ with a scalar index returns window (1, window_size) and scalar
    target."""
    window_size = 4
    with _patch_run_discharge(n_t=12):
        source = BatterySimulationTimeWindowSource(
            simulator_config={},
            n_histories=1,
            window_size=window_size,
        )

    window, target = source[0]
    assert window.shape == (1, window_size)
    assert target.shape == ()


def test_time_window_source_getitem_batch_shape() -> None:
    """__getitem__ with a slice returns windows (batch, 1, window_size) and targets
    (batch,)."""
    window_size = 4
    batch_size = 3
    with _patch_run_discharge(n_t=12):
        source = BatterySimulationTimeWindowSource(
            simulator_config={},
            n_histories=1,
            window_size=window_size,
        )

    windows, targets = source[:batch_size]
    assert windows.shape == (batch_size, 1, window_size)
    assert targets.shape == (batch_size,)


def test_time_window_source_soc_range() -> None:
    """All sampled SoC₀ values must lie within the requested range."""
    soc_range = (0.2, 0.8)
    with _patch_run_discharge(n_t=10):
        source = BatterySimulationTimeWindowSource(
            simulator_config={},
            n_histories=50,
            window_size=3,
            soc_range=soc_range,
        )
    assert all(soc_range[0] <= s <= soc_range[1] for s in source.soc_0s)


def test_time_window_source_short_histories_included() -> None:
    """Histories shorter than window_size are padded, not skipped."""
    voltage_short = np.array([4.0, 3.5])

    def _stub(_config, _soc_0):
        return voltage_short.copy(), 1.0

    with patch("qmodem.data._run_discharge", side_effect=_stub):
        source = BatterySimulationTimeWindowSource(
            simulator_config={},
            n_histories=3,
            window_size=5,
        )
    # Each short history should produce at least one window
    assert len(source) >= 3
