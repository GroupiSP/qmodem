"""Integration test for BatterySimulationTimeWindowSource with Google Grain."""

from unittest.mock import Mock

import numpy as np
import pytest
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem.data import BatterySimulationTimeWindowSource


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


def test_dataloader_integration(mock_simulator):
    """Test that BatterySimulationTimeWindowSource works with Grain DataLoader."""
    # Create the data source
    source = BatterySimulationTimeWindowSource(
        mock_simulator, window_size=3, stride=1, normalize=False
    )

    # Create sampler
    sampler = IndexSampler(num_records=len(source), num_epochs=1, shuffle=True, seed=42)

    # Create DataLoader with batching
    batch_size = 4
    dataloader = DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=batch_size)],
        worker_count=0,
    )

    # Iterate through batches
    batches = list(dataloader)
    assert len(batches) > 0

    # Check first batch
    first_batch = batches[0]
    windows, targets = first_batch
    assert windows.shape[0] <= batch_size  # Batch dimension
    assert windows.shape[1] == 1  # Height dimension from (1, window_size)
    assert windows.shape[2] == 3  # window_size
    assert targets.shape[0] <= batch_size


def test_dataloader_all_samples_retrieved(mock_simulator):
    """Test that DataLoader retrieves all samples from the source."""
    source = BatterySimulationTimeWindowSource(
        mock_simulator, window_size=2, stride=1, normalize=False
    )

    sampler = IndexSampler(num_records=len(source), num_epochs=1, shuffle=False, seed=0)

    batch_size = 3
    dataloader = DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=batch_size)],
        worker_count=0,
    )

    total_samples = 0
    for batch in dataloader:
        windows, targets = batch
        total_samples += windows.shape[0]

    assert total_samples == len(source)


def test_dataloader_with_normalization(mock_simulator):
    """Test that DataLoader works with normalized data."""
    source = BatterySimulationTimeWindowSource(
        mock_simulator, window_size=2, stride=1, normalize=True
    )

    sampler = IndexSampler(num_records=len(source), num_epochs=1, shuffle=False, seed=0)

    dataloader = DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=2)],
        worker_count=0,
    )

    # Get first batch and check normalization
    first_batch = next(iter(dataloader))
    windows, targets = first_batch

    # Targets should be normalized (all <= 1.0)
    assert np.all(targets <= 1.0)
