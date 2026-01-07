from typing import SupportsIndex

import lib_eod_simulation as les
import numpy as np


class BatterySimulationSingleTimeSource:
    def __init__(self, simulator: les.SimulatorSimple | les.SimulatorComplete) -> None:
        """Runs and access to battery simulation data.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source. It must have `N_simu=1` (only deterministic case, work in progress
                to extend).
        """
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
    print("Hello from qmodem!")
