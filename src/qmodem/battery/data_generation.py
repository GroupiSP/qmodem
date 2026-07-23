from __future__ import annotations

import dataclasses
from enum import StrEnum, auto

import pandas as pd
import simbat as sb


class VOCModel(StrEnum):
    BUSTOS_BAEZA = auto()


class ECMModel(StrEnum):
    THEVENIN_ZERO_ORDER = auto()


@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    voc_model: VOCModel = VOCModel.BUSTOS_BAEZA
    ecm_model: ECMModel = ECMModel.THEVENIN_ZERO_ORDER
    ecm_model_params: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"r0": 0.1},
    )
    battery_nominal_capacity: float = 10080.0  # in Coulombs
    dt: float = 60.0
    v_cutoff: float = 2.5  # in Volts
    n_histories_train: int = 50
    n_histories_val: int = 20
    n_histories_test: int = 10
    process_noise_std: float = 5e-3
    measurement_noise_std: float = 5e-3
    soc_range_train_val: tuple[float, float] = (0.05, 1.0)
    train_seed: int = 42
    test_seed: int = 123


def _modify_dataframe(df: pd.DataFrame, run_id: int) -> None:
    df.drop(
        columns=["rul_probability", "eod_reached_sim_0"], inplace=True
    )  # Drop the RUL probability column
    df.rename(
        columns={"time": "time", "soc_sim_0": "soc", "voltage_sim_0": "voltage"},
        inplace=True,
    )
    df.insert(0, "run_id", run_id)  # Add a run_id column for tracking
    return None


def write_histories(config: sb.SimulationConfig, n_histories: int) -> pd.DataFrame:
    out_df = pd.DataFrame(columns=["run_id", "time", "soc", "voltage"])

    for i in range(n_histories):
        result = sb.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()

        # Modify the dataframe and append it to the output one
        _modify_dataframe(df, i)

        out_df = pd.concat([out_df, df], ignore_index=True)

    return out_df
