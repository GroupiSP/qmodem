import json

import lib_eod_simulation as les
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import BatterySimulationSingleTimeSource


def get_battery_dataloader(N_simu: int = 1) -> DataLoader:
    with open("./battery_sim_config.json") as fp:
        sim_config = json.load(fp)

    I_discharge = les.ConstantCurrentDischarge(sim_config["I_const_discharge"])

    sim = les.SimulatorSimple(
        N_simu,
        sim_config["v_cut"],
        sim_config["SoC"],
        I_discharge,
        sim_config["model_config"],
    )

    source = BatterySimulationSingleTimeSource(sim)
    sampler = IndexSampler(num_records=len(source), shuffle=True, seed=0)
    return DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=10)],
        worker_count=0,
    )


def main() -> None:
    dataloader = get_battery_dataloader()

    print(next(iter(dataloader)))


if __name__ == "__main__":
    main()
