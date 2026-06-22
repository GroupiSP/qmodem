"""Tests for qmodem.data module."""

from __future__ import annotations

import numpy as np

from qmodem.data import (
    _make_windows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
# Tests for _make_windows
# ---------------------------------------------------------------------------


class TestMakeWindows:
    def test_basic_shape(self) -> None:
        voltage = np.linspace(4.2, 3.0, 10)
        ruls = np.linspace(5.0, 0.0, 10)
        windows, targets = _make_windows(voltage, ruls, window_size=3, stride=1)
        for w in windows:
            assert w.shape == (3, 1)
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
        assert w2[0].size < w1[0].size
