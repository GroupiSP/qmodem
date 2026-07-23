from __future__ import annotations

import functools
import pathlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import simbat as sb

from qmodem.battery.policies import (
    VariableDischargeCurrentPolicy,
    plot_current_profile,
)
from qmodem.tracking import MLFlowSetup, track_mlflow
from scripts.battery_multiple_scenarios.commons import (
    RAW_DATA_DIR,
)


class ProcessNoiseDistribution(StrEnum):
    NORMAL = auto()
    UNIFORM = auto()
    ZERO = auto()


class VOCModel(StrEnum):
    BUSTOS_BAEZA = auto()


class ECMModel(StrEnum):
    THEVENIN_ZERO_ORDER = auto()


dist_name_to_params = {
    ProcessNoiseDistribution.NORMAL: lambda loc, scale: {"loc": loc, "scale": scale},
    ProcessNoiseDistribution.UNIFORM: lambda low, high: {"low": low, "high": high},
    ProcessNoiseDistribution.ZERO: lambda: {},
}

ecm_model_name_to_params = {
    ECMModel.THEVENIN_ZERO_ORDER: lambda r0: {"r0": r0},
}

constant_cruise_policy = VariableDischargeCurrentPolicy(
    current_values=[-4.0, -1.0],
    time_values=[0.0, 600.0],
)
variable_cruise_policy = VariableDischargeCurrentPolicy(
    current_values=[-4.0, -1.0, -2.0, -1.0],
    time_values=[0.0, 600.0, 1800.0, 4000.0],
)


@dataclass(frozen=True)
class Hyperparameters:
    voc_model: VOCModel = VOCModel.BUSTOS_BAEZA
    ecm_model: ECMModel = ECMModel.THEVENIN_ZERO_ORDER
    ecm_model_params: dict[str, float] = field(
        default_factory=lambda: ecm_model_name_to_params[ECMModel.THEVENIN_ZERO_ORDER](
            r0=0.1
        )
    )
    battery_nominal_capacity: float = 10080.0  # in Coulombs
    dt: float = 60.0
    v_cutoff: float = 2.5  # in Volts
    n_histories_train: int = 50
    n_histories_val: int = 20
    n_histories_test: int = 10
    process_noise_distribution: ProcessNoiseDistribution = (
        ProcessNoiseDistribution.NORMAL
    )
    measurement_noise_distribution: ProcessNoiseDistribution = (
        ProcessNoiseDistribution.ZERO
    )
    # TODO: noise parameter tracking can be improved by using `dist_name_to_params`. But it
    # is not a prio right now.
    process_noise_loc: float = 0.0
    process_noise_std: float = 5e-3
    measurement_noise_param: Any = None
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


def policy_choice_distribution(*, rng: np.random.Generator) -> int:
    return rng.choice([0, 1], p=[0.7, 0.3])


def process_noise_distribution(
    *,
    rng: np.random.Generator,
    process_noise_std: float,
    process_noise_loc: float = 0.0,
) -> float:
    return rng.normal(loc=process_noise_loc, scale=process_noise_std)


def measurement_noise_distribution() -> float:
    return 0.0


def make_simulator_config(
    rng: np.random.Generator, hp: Hyperparameters
) -> sb.SimulationConfig:
    """Creates a simulation configuration for the battery discharge simulation.

    Args:
        rng: A NumPy random number generator for reproducibility.
        hp: Hyperparameters for the simulation.
    """
    return sb.SimulationConfig(
        current_policies=[constant_cruise_policy, variable_cruise_policy],
        policy_choice_distribution=functools.partial(
            policy_choice_distribution, rng=rng
        ),
        process_noise_distribution=functools.partial(
            process_noise_distribution,
            rng=rng,
            process_noise_std=hp.process_noise_std,
        ),
        measurement_noise_distribution=measurement_noise_distribution,
        dt=hp.dt,
        soc_0=1.0,
    )


def write_histories(
    rng: np.random.Generator, hp: Hyperparameters, n_histories: int
) -> pd.DataFrame:
    out_df = pd.DataFrame(columns=["run_id", "time", "soc", "voltage"])

    for i in range(n_histories):
        config = make_simulator_config(rng=rng, hp=hp)
        result = sb.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()

        # Modify the dataframe and append it to the output one
        _modify_dataframe(df, i)

        out_df = pd.concat([out_df, df], ignore_index=True)

    return out_df


def save_dataframe_to_file(df: pd.DataFrame, path: pathlib.Path) -> None:
    df.to_csv(path, index=False)


def track_dataframe(df: pd.DataFrame, name: str, context: str) -> None:
    dataset = mlflow.data.from_pandas(df, name=name)
    mlflow.log_input(dataset=dataset, context=context)


def main() -> None:
    hp = Hyperparameters()

    # MLFlow setup
    run_tags = {
        "case_study": "battery",
        "stage": "data_generation",
    }
    tracking_setup = MLFlowSetup(
        experiment_name="refactoring_jul_2026",
        run_name="gen_multiple",
        tags=run_tags,
    )
    with track_mlflow(tracking_setup):
        train_rng = np.random.default_rng(seed=hp.train_seed)
        test_rng = np.random.default_rng(seed=hp.test_seed)

        mlflow.log_params(asdict(hp))

        train_df = write_histories(
            train_rng, hp, n_histories=hp.n_histories_train + hp.n_histories_val
        )
        test_df = write_histories(test_rng, hp, n_histories=hp.n_histories_test)

        save_dataframe_to_file(train_df, path=RAW_DATA_DIR / "train.csv")
        save_dataframe_to_file(test_df, path=RAW_DATA_DIR / "test.csv")

        track_dataframe(train_df, name="battery_train", context="train")
        track_dataframe(test_df, name="battery_test", context="test")

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        plot_current_profile(ax=axs[0], policy=constant_cruise_policy)
        axs[0].set_title("Constant Cruise Policy")

        plot_current_profile(ax=axs[1], policy=variable_cruise_policy)
        axs[1].set_title("Variable Cruise Policy")
        mlflow.log_figure(fig, artifact_file="current_profiles.png")

        res_for_plots = sb.simulate_constant_capacity_simple(
            n_sim=1_000,
            config=make_simulator_config(
                rng=np.random.default_rng(seed=hp.train_seed + 1), hp=hp
            ),
        )

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        sb.plot_rul_results(ax=axs[0], results=[res_for_plots])
        sb.plot_rul_bars(ax=axs[1], results=[res_for_plots])
        sb.plot_voltage_results(ax=axs[2], results=[res_for_plots])
        sb.plot_soc_results(ax=axs[3], results=[res_for_plots])

        mlflow.log_figure(fig, artifact_file="discharge_results.png")

        # Make another plot for the voltage discharge of the first 10 test cases
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
        for i in range(10):
            ax = axs[i // 5, i % 5]
            test_case_df = test_df[test_df["run_id"] == i]
            ax.plot(test_case_df["time"], test_case_df["voltage"])
            ax.set_title(f"Test Case {i}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Voltage (V)")
            ax.grid()
        fig.tight_layout()
        mlflow.log_figure(fig, artifact_file="test_case_voltage_discharge.png")


if __name__ == "__main__":
    main()
