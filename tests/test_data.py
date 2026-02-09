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
    simulator.t_eods = np.array([1.0])
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
    assert np.isclose(ruls[-1], 0.0)


class TestBatterySimulationTimeWindowSourceInit:
    def test_init_single_simulation(self, mock_simulator):
        """Test initialization with valid single simulation."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        assert source.window_size == 3
        assert source.stride == 1
        assert source.N_t == 10
        mock_simulator.simulate.assert_called_once()

    def test_init_multiple_simulations_raises_error(self, mock_simulator_multiple):
        """Test that multiple simulations raise ValueError."""
        with pytest.raises(ValueError, match="only supports a single simulation"):
            BatterySimulationTimeWindowSource(mock_simulator_multiple, window_size=3)

    def test_init_stores_discharge_voltage(self, mock_simulator):
        """Test that discharge voltage is properly stored."""
        source = BatterySimulationTimeWindowSource(mock_simulator, window_size=2)
        assert source.discharge_voltage.shape == (10,)
        np.testing.assert_array_almost_equal(
            source.discharge_voltage, mock_simulator.v_memo.T[0]
        )

    def test_init_calculates_ruls(self, mock_simulator):
        """Test that RULs are properly calculated."""
        source = BatterySimulationTimeWindowSource(mock_simulator, window_size=2)
        assert source.ruls.shape == (10,)
        assert source.ruls[0] == mock_simulator.t_eods
        assert source.ruls[-1] == 0.0

    def test_init_discharge_voltage_and_ruls_consistent(self, mock_simulator):
        """Test that discharge voltage and RULs are consistent in length."""
        source = BatterySimulationTimeWindowSource(mock_simulator, window_size=2)
        assert len(source.discharge_voltage) == len(source.ruls)


class TestBatterySimulationTimeWindowSourceIter:
    def test_iter_returns_windows_and_targets(self, mock_simulator):
        """Test iteration yields windows and targets."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        records = list(source)
        assert len(records) == 8  # N_t - window_size + 1 = 10 - 3 + 1 = 8
        for window, target in records:
            assert window.shape == (1, 3)
            assert target.shape == ()

    def test_iter_window_shape(self, mock_simulator):
        """Test that window has correct shape."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=4, stride=1
        )
        first_window, _ = next(iter(source))
        assert first_window.shape == (1, 4)

    def test_iter_stride_works(self, mock_simulator):
        """Test that stride parameter works correctly."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=2, stride=3
        )
        records = list(source)
        # range(0, 10 - 2 + 1, 3) = range(0, 9, 3) = [0, 3, 6]
        assert len(records) == 3

    def test_iter_last_window_has_zero_rul(self, mock_simulator):
        """Test that the last window is assigned RUL of 0."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        records = list(source)
        _, last_target = records[-1]
        assert last_target == 0.0

    def test_iter_non_last_windows_have_correct_targets(self, mock_simulator):
        """Test that non-last windows have correct RUL targets."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=2, stride=1
        )
        records = list(source)
        # First window ends at index 2, target should be ruls[2]
        _, first_target = records[0]
        assert float(first_target) == float(source.ruls[2])

    def test_iter_window_content_correct(self, mock_simulator):
        """Test that window content matches expected discharge voltage values."""
        source = BatterySimulationTimeWindowSource(
            mock_simulator, window_size=3, stride=1
        )
        first_window, _ = next(iter(source))
        expected = mock_simulator.v_memo[0:3].reshape(1, -1)
        np.testing.assert_array_almost_equal(first_window, expected)
