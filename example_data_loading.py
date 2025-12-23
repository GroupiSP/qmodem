from typing import SupportsIndex

import lib_eod_simulation as les
import numpy as np
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch


class BatterySimulationSingleTimeSource:
    def __init__(self, simulator: les.SimulatorSimple | les.SimulatorComplete) -> None:
        """Runs and access to battery simulation data.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source. It must have `N_simu=1` (only deterministic case, work in progress
                to extend).
        """
        if simulator.N_simu != 1:
            raise ValueError("We only support one simulation at the moment.")

        simulator.simulate()

        self.discharge_voltage = simulator.v_memo.ravel()
        self.times = np.arange(len(self.discharge_voltage)) * simulator.batt.dt
        self.ruls = self.times[-1] - self.times

    def __len__(self) -> int:
        """Number of records in the dataset."""
        return len(self.discharge_voltage)

    def __getitem__(self, record_key: SupportsIndex) -> np.ndarray:
        """Retrieves record for the given record_key."""
        return self.discharge_voltage[record_key], self.ruls[record_key]


def main() -> None:
    model_config = {
        "Q": 2.8 * 3600,
        "R": 0.1,
        "dt": 10,  # (indirectly) the grid resolution.
        "voc_params": {
            "vL": 1.35531394,
            "v0": 4.12017677,
            "gamma": 0.13286143,
            "alpha": 0.16945463,
            "beta": 2.34538224,
        },
        "omega_std": 0.0,  # Process noise (play with this)
        "eta_std": 0.0,  # Voltage sensor noise (play with this)
    }
    v_cut = 2.5  # Cutoff voltage of the battery
    SoC = 1  # Initial state of charge

    I_discharge = les.ConstantCurrentDischarge(
        -2.8 * 0.75
    )  # Constant current discharge policy (load)

    sim = les.SimulatorSimple(1, v_cut, SoC, I_discharge, model_config)

    source = BatterySimulationSingleTimeSource(sim)
    sampler = IndexSampler(num_records=len(source), shuffle=True, seed=0)
    dataloader = DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=10)],
        worker_count=0,
    )

    print(next(iter(dataloader)))


if __name__ == "__main__":
    main()
