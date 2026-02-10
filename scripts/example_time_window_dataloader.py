"""Example of using BatterySimulationTimeWindowSource with Google Grain DataLoader.

This demonstrates how to create time-windowed battery simulation data and use it with
batching for training.
"""

import json

import lib_eod_simulation as les
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem.data import BATT_CONFIG_PATH, BatterySimulationTimeWindowSource


def main():
    """Demonstrates BatterySimulationTimeWindowSource with Grain DataLoader."""
    # Load battery configuration
    with open(BATT_CONFIG_PATH) as f:
        batt_config = json.load(f)

    # Create discharge policy and battery
    discharge_policy = les.ConstantCurrentDischarge(-2.8)
    battery = les.BatteryModel(batt_config)

    # Create simulator configuration (single simulation for time window source)
    simulator_config = {
        "N_simu": 1,  # Must be 1 for time window source
        "v_cut": 2.5,
        "SoC_0": 1.0,
        "dt": 10.0,
        "omega_std": 1e-3,
        "eta_std": 1e-2,
        "I": discharge_policy,
        "battery": battery,
    }

    simulator = les.SimulatorSimple(simulator_config)

    # Create time window data source
    window_size = 50  # Number of time steps in each window
    stride = 10  # Stride between windows
    normalize = True  # Normalize RUL values

    data_source = BatterySimulationTimeWindowSource(
        simulator=simulator,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
    )

    print(f"Created data source with {len(data_source)} windows")
    print(f"Window shape: (1, {window_size})")
    print(f"Max RUL value: {data_source.y_max:.4f}")

    # Create sampler for training
    sampler = IndexSampler(
        num_records=len(data_source),
        num_epochs=1,
        shuffle=True,
        seed=42,
    )

    # Create DataLoader with batching
    batch_size = 32
    dataloader = DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[Batch(batch_size=batch_size)],
        worker_count=0,
    )

    # Iterate through batches
    print(f"\nIterating through batches (batch_size={batch_size}):")
    for i, (windows, targets) in enumerate(dataloader):
        print(
            f"Batch {i}: windows shape={windows.shape}, targets shape={targets.shape}"
        )
        if i >= 2:  # Show first 3 batches
            break

    # Show individual sample access
    print("\nRandom access to individual samples:")
    for idx in [0, len(data_source) // 2, len(data_source) - 1]:
        window, target = data_source[idx]
        print(f"Sample {idx}: window shape={window.shape}, target={target:.4f}")


if __name__ == "__main__":
    main()
