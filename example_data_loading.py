import json
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
    with open("./battery_sim_config.json") as fp:
        sim_config = json.load(fp)

    I_discharge = les.ConstantCurrentDischarge(sim_config["I_const_discharge"])

    sim = les.SimulatorSimple(
        1,
        sim_config["v_cut"],
        sim_config["SoC"],
        I_discharge,
        sim_config["model_config"],
    )

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
