from __future__ import annotations

import json
from pathlib import Path
from typing import SupportsIndex

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import numpy as np

BATT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "battery_config.json"


class BatterySimulationSource:
    def __init__(self, simulator: les.SimulatorSimple | les.SimulatorComplete) -> None:
        """Runs and access to battery simulation data.

        Args:
            simulator (les.SimulatorSimple | les.SimulatorComplete): the simulator from
                lib_eod_simulation. It needs to be configured outside of this data
                source. It must have `N_simu=1` (only deterministic case, work in progress
                to extend).
        """
        simulator.simulate()

        # Transpose for convenience. Shape=(N_simu, N_t).
        discharge_voltage_per_sim: np.ndarray = simulator.v_memo.T
        N_t = discharge_voltage_per_sim.shape[1]

        self.discharge_voltage = jnp.array(
            discharge_voltage_per_sim.flatten().reshape(-1, 1)
        )
        ruls = np.empty(shape=(simulator.N_simu * N_t))

        for i in range(simulator.N_simu):
            ruls[i * N_t : (i + 1) * N_t] = np.clip(
                simulator.t_eods[i] - np.arange(N_t) * simulator.dt,
                a_min=0.0,
                a_max=None,
            )  # clipping ensures that the failed particles have RUL=0. after their time of failure

        self.ruls = jnp.array(ruls)

    def __len__(self) -> int:
        """Number of records in the dataset."""
        return len(self.ruls)

    def __getitem__(self, record_key: SupportsIndex) -> tuple[jax.Array, float]:
        """Retrieves record for the given record_key."""
        return self.discharge_voltage[record_key], self.ruls[record_key]


def make_battery_data(
    N_simu: int = 1,
) -> tuple[les.SimulatorSimple, BatterySimulationSource]:
    """Makes the Grain data source for the battery simulator. Assumes a constant current
    policy.

    Args:
        N_simu (int, optional): Number of MC simulations of the battery discharge. Defaults to 1.

    Returns:
        BatterySimulationSource: the Grain battery data-source.
    """
    with open(BATT_CONFIG_PATH) as fp:
        sim_config = json.load(fp)

    I_discharge = les.ConstantCurrentDischarge(sim_config["I_const_discharge"])

    sim = les.SimulatorSimple(
        N_simu,
        sim_config["v_cut"],
        sim_config["SoC"],
        I_discharge,
        sim_config["model_config"],
    )

    return sim, BatterySimulationSource(sim)
