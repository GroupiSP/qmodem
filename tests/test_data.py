from unittest.mock import Mock

import numpy as np
import pytest

from qmodem.data import BatterySimulationTimeWindowSource, _back_calculate_rul_linear


@pytest.fixture
def mock_simulator():
    """Create a mock simulator with single simulation."""
    simulator = Mock()
    simulator.N_simu = 1
    simulator.dt = 0.1
    simulator.t_eods = np.array(
        [10.0]
    )  # Changed from 1.0 for clearer normalization tests
    # Create discharge voltage data: shape (1, N_t) after transpose
    simulator.v_memo = np.array(
        [4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3]
    ).reshape(-1, 1)
    simulator.simulate = Mock()
    return simulator


@pytest.fixture
def mock_simulator_multiple():
    """Create a mock simulator with multiple simulations."""
    simulator = Mock()
    simulator.N_simu = 2
    return simulator


def test_back_calculate_rul_linear():
    """Test the _back_calculate_rul_linear function."""
    t_eod = 10.0
    ruls = _back_calculate_rul_linear(t_eod=t_eod, N_t=100)
    assert len(ruls) == 100
    assert np.isclose(ruls[0], 10.0)


class TestBatterySimulationTimeWindowSourceInit:
    def test_init_single_simulation(self, mock_simulator):
        """Test initialization with valid single simulation."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        assert len(source) == 8  # (10 - 3) // 1 + 1 = 8
        mock_simulator.simulate.assert_called_once()

    def test_init_multiple_simulations_raises_error(self, mock_simulator_multiple):
        """Test that multiple simulations raise ValueError."""
        with pytest.raises(ValueError, match="only supports a single simulation"):
            BatterySimulationTimeWindowSource(mock_simulator_multiple, window_size=3)

    def test_init_window_size_too_large_raises_error(self, mock_simulator):
        """Test that window_size > N_t raises ValueError."""
        with pytest.raises(ValueError, match="window_size"):
            BatterySimulationTimeWindowSource(mock_simulator, window_size=20)

    def test_init_stores_windows_and_targets(self, mock_simulator):
        """Test that windows and targets are properly stored."""
        source = BatterySimulationTimeWindowSource(mock_simulator, window_size=2)
        assert source.X.shape == (9, 1, 2)  # (num_windows, 1, window_size)
        assert source.y.shape == (9,)  # num_windows

    def test_init_y_max_calculated(self, mock_simulator):
        """Test that y_max is properly calculated."""
        source = BatterySimulationTimeWindowSource(mock_simulator, window_size=2)
        assert source.y_max > 0

    def test_init_normalization_disabled(self, mock_simulator):
        """Test that normalization can be disabled."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=2, normalize=False
        )
        # y_max should be close to t_eod (10.0) when not normalized
        assert float(source.y_max) > 5.0  # Should be close to 10.0
        assert float(source.y_max) <= 10.0

    def test_init_normalization_enabled(self, mock_simulator):
        """Test that normalization works correctly."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=2, normalize=True
        )
        # All RUL values should be <= 1.0 when normalized
        assert float(np.max(source.y)) <= 1.0


class TestBatterySimulationTimeWindowSourceAccess:
    def test_len_returns_num_windows(self, mock_simulator):
        """Test __len__ returns correct number of windows."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        assert len(source) == 8  # (10 - 3) // 1 + 1 = 8

    def test_len_with_stride(self, mock_simulator):
        """Test __len__ with stride parameter."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=2, stride=3
        )
        # (10 - 2) // 3 + 1 = 8 // 3 + 1 = 2 + 1 = 3
        assert len(source) == 3

    def test_getitem_returns_window_and_target(self, mock_simulator):
        """Test __getitem__ returns correct window and target."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        window, target = source[0]
        assert window.shape == (1, 3)
        assert target.shape == ()

    def test_getitem_window_content_correct(self, mock_simulator):
        """Test that window content matches expected discharge voltage values."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        first_window, _ = source[0]
        expected = mock_simulator.v_memo[0:3].reshape(1, -1)
        np.testing.assert_array_almost_equal(first_window, expected)

    def test_getitem_last_window_has_zero_rul(self, mock_simulator):
        """Test that the last window is assigned RUL of 0."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        _, last_target = source[len(source) - 1]
        assert float(last_target) == 0.0

    def test_getitem_multiple_indices(self, mock_simulator):
        """Test __getitem__ with multiple indices."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=2, stride=1
        )
        # Test that we can access all windows
        for i in range(len(source)):
            window, target = source[i]
            assert window.shape == (1, 2)
            assert target.shape == ()

    def test_getitem_stride_correct_windows(self, mock_simulator):
        """Test that stride creates correct windows."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=2, stride=3
        )
        # First window: indices 0-1
        # Second window: indices 3-4
        # Third window: indices 6-7
        assert len(source) == 3
        first_window, _ = source[0]
        expected_first = mock_simulator.v_memo[0:2].reshape(1, -1)
        np.testing.assert_array_almost_equal(first_window, expected_first)

        second_window, _ = source[1]
        expected_second = mock_simulator.v_memo[3:5].reshape(1, -1)
        np.testing.assert_array_almost_equal(second_window, expected_second)
