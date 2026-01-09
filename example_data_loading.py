import json

import lib_eod_simulation as les
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import BatterySimulationSource


def quickstart_dataloader(N_simu: int = 1, batch_size: int = 10) -> DataLoader:
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

    source = BatterySimulationSource(sim)
    sampler = IndexSampler(num_records=len(source), num_epochs=1, shuffle=True, seed=0)
    return DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=batch_size)],
        worker_count=0,
    )


def main() -> None:
    # Run iid simulations for training and testing
    dataloader_train = quickstart_dataloader(N_simu=5, batch_size=10)
    dataloader_test = quickstart_dataloader(N_simu=1, batch_size=5)

    print(next(iter(dataloader_train)))
    print(next(iter(dataloader_test)))


if __name__ == "__main__":
    main()
