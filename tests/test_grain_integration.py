"""Integration test for BatterySimulationTimeWindowSource with Google Grain."""

from unittest.mock import patch

import numpy as np
import pytest
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem.data import BatterySimulationTimeWindowSource


def _stub_run_discharge(_config, _soc_0):
    """Deterministic stub returning a 10-step discharge history."""
    voltage = np.array([4.2, 4.1, 4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3])
    return voltage.copy(), 10.0


@pytest.fixture
def _patch_discharge():
    """Patch ``_run_discharge`` so no real simulator is needed."""
    with patch("qmodem.data._run_discharge", side_effect=_stub_run_discharge):
        yield


@pytest.mark.usefixtures("_patch_discharge")
def test_dataloader_integration():
    """Test that BatterySimulationTimeWindowSource works with Grain DataLoader."""
    source = BatterySimulationTimeWindowSource(
        simulator_config={}, n_histories=1, window_size=3, stride=1, normalize=False
    )

    sampler = IndexSampler(num_records=len(source), num_epochs=1, shuffle=True, seed=42)

    batch_size = 4
    dataloader = DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=batch_size)],
        worker_count=0,
    )

    batches = list(dataloader)
    assert len(batches) > 0

    first_batch = batches[0]
    windows, targets = first_batch
    assert windows.shape[0] <= batch_size
    assert windows.shape[1] == 1  # Height dimension from (1, window_size)
    assert windows.shape[2] == 3  # window_size
    assert targets.shape[0] <= batch_size


@pytest.mark.usefixtures("_patch_discharge")
def test_dataloader_all_samples_retrieved():
    """Test that DataLoader retrieves all samples from the source."""
    source = BatterySimulationTimeWindowSource(
        simulator_config={}, n_histories=1, window_size=2, stride=1, normalize=False
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


@pytest.mark.usefixtures("_patch_discharge")
def test_dataloader_with_normalization():
    """Test that DataLoader works with normalized data."""
    source = BatterySimulationTimeWindowSource(
        simulator_config={}, n_histories=1, window_size=2, stride=1, normalize=True
    )

    sampler = IndexSampler(num_records=len(source), num_epochs=1, shuffle=False, seed=0)

    dataloader = DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=2)],
        worker_count=0,
    )

    first_batch = next(iter(dataloader))
    windows, targets = first_batch

    assert np.all(targets <= 1.0)
