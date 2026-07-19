from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import simbat as sb

from qmodem.tracking import MLFlowSetup, track_mlflow


class VariableDischargeCurrentPolicy:
    def __init__(self, current_values: list[float], time_values: list[float]) -> None:
        self.current_values = current_values
        self.time_values = time_values

    def __call__(self, soc: float, t: float) -> float:
        """Returns the current values at time `t` for the given SoC values."""
        for i in range(len(self.time_values) - 1):
            if self.time_values[i] <= t < self.time_values[i + 1]:
                return self.current_values[i]
        return self.current_values[
            -1
        ]  # Return the last current value if t exceeds the last time value


class ProcessNoiseDistribution(StrEnum):
    NORMAL = auto()
    UNIFORM = auto()
    ZERO = auto()


class CurrentPolicy(StrEnum):
    CONSTANT = auto()
    VARIABLE = auto()


class VOCModel(StrEnum):
    BUSTOS_BAEZA = auto()


class ECMModel(StrEnum):
    THEVENIN_ZERO_ORDER = auto()


dist_name_to_params = {
    ProcessNoiseDistribution.NORMAL: lambda loc, scale: {"loc": loc, "scale": scale},
    ProcessNoiseDistribution.UNIFORM: lambda low, high: {"low": low, "high": high},
    ProcessNoiseDistribution.ZERO: lambda: {},
}

name_to_policy = {
    # Current values in Amperes
    CurrentPolicy.CONSTANT: sb.simulate.ConstantCurrentDischarge(
        current_value=-2.8 * 0.75
    ),
    CurrentPolicy.VARIABLE: VariableDischargeCurrentPolicy(
        current_values=[-2.0, -1.0, -4.0, -2.0, -3.0],
        time_values=[0.0, 600.0, 900.0, 1800.0, 3000.0],
    ),
}

ecm_model_name_to_params = {
    ECMModel.THEVENIN_ZERO_ORDER: lambda r0: {"r0": r0},
}


@dataclass(frozen=True)
class Hyperparameters:
    current_policy: CurrentPolicy = CurrentPolicy.CONSTANT
    voc_model: VOCModel = VOCModel.BUSTOS_BAEZA
    ecm_model: ECMModel = ECMModel.THEVENIN_ZERO_ORDER
    ecm_model_params: dict[str, float] = field(
        default_factory=lambda: ecm_model_name_to_params[ECMModel.THEVENIN_ZERO_ORDER](
            r0=0.1
        )
    )
    battery_nominal_capacity: float = 10080.0  # in Coulombs
    dt: float = 20.0
    v_cutoff: float = 2.5  # in Volts
    n_histories_train: int = 100
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
    process_noise_std: float = 3e-3
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


def generate_train(rng: np.random.Generator, hp: Hyperparameters) -> pd.DataFrame:
    soc_0s = rng.uniform(
        low=hp.soc_range_train_val[0],
        high=hp.soc_range_train_val[1],
        size=hp.n_histories_train + hp.n_histories_val,
    )

    out_df = pd.DataFrame(columns=["run_id", "time", "soc", "voltage"])

    for i, soc_0 in enumerate(soc_0s):
        config = sb.SimulationConfig(
            current_policy=name_to_policy[hp.current_policy],
            process_noise_distribution=lambda: rng.normal(
                loc=hp.process_noise_loc, scale=hp.process_noise_std
            ),
            measurement_noise_distribution=lambda: 0.0,
            dt=hp.dt,
            soc_0=soc_0,
        )
        result = sb.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()

        # Modify the dataframe and append it to the output one
        _modify_dataframe(df, i)

        out_df = pd.concat([out_df, df], ignore_index=True)

    return out_df


def generate_test(rng: np.random.Generator, hp: Hyperparameters) -> pd.DataFrame:
    out_df = pd.DataFrame(columns=["run_id", "time", "soc", "voltage"])

    for i in range(hp.n_histories_test):
        config = sb.SimulationConfig(
            current_policy=name_to_policy[hp.current_policy],
            process_noise_distribution=lambda: rng.normal(
                loc=hp.process_noise_loc, scale=hp.process_noise_std
            ),
            measurement_noise_distribution=lambda: 0.0,
            dt=hp.dt,
            soc_0=1.0,
        )
        result = sb.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()

        _modify_dataframe(df, i)

        out_df = pd.concat([out_df, df], ignore_index=True)

    return out_df


def plot_current_policy(ax: plt.Axes, hp: Hyperparameters) -> None:
    t_grid = np.linspace(0, 5000, 100)
    current_values = [name_to_policy[hp.current_policy](soc=None, t=t) for t in t_grid]
    ax.plot(t_grid, current_values)
    ax.set_xlabel("Time")
    ax.set_ylabel("Current")
    ax.grid()


def run_sims_from_tzero(
    rng: np.random.Generator, hp: Hyperparameters
) -> sb.SimulationResult:
    config = sb.SimulationConfig(
        current_policy=name_to_policy[hp.current_policy],
        process_noise_distribution=lambda: rng.normal(
            loc=hp.process_noise_loc, scale=hp.process_noise_std
        ),
        measurement_noise_distribution=lambda: 0.0,
        dt=hp.dt,
        soc_0=1.0,
    )
    return sb.simulate_constant_capacity_simple(n_sim=1_000, config=config)


def save_dataframe_to_file(df: pd.DataFrame, path: pathlib.Path) -> None:
    df.to_csv(path, index=False)


def track_dataframe(df: pd.DataFrame, name: str, context: str) -> None:
    dataset = mlflow.data.from_pandas(df, name=name)
    mlflow.log_input(dataset=dataset, context=context)


def main() -> None:
    """Single access point to generate the training, validation and test discharge
    histories and save them to disk.

    Train/validation data generation:
    - 100 discharge histories for training, 20 for validation.
    - Initial SoC sampled uniformly from [0.05, 1.0].
    - Same RNG for reproducibility.

    Test data generation:
    - 10 test cases.
    - Different RNG to ensure independent test data.
    """
    BATTERY_DATA_DIR = (
        pathlib.Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "battery"
    )

    hp = Hyperparameters(
        current_policy=CurrentPolicy.VARIABLE,
    )

    # MLFlow setup
    run_tags = {
        "case_study": "battery",
        "stage": "data_generation",
    }
    tracking_setup = MLFlowSetup(
        experiment_name="reliability_study",
        run_name="generate_discharge_histories",
        tags=run_tags,
    )
    with track_mlflow(tracking_setup):
        train_rng = np.random.default_rng(seed=hp.train_seed)
        test_rng = np.random.default_rng(seed=hp.test_seed)

        mlflow.log_params(asdict(hp))

        train_df = generate_train(train_rng, hp)
        test_df = generate_test(test_rng, hp)

        save_dataframe_to_file(train_df, path=BATTERY_DATA_DIR / "train.csv")
        save_dataframe_to_file(test_df, path=BATTERY_DATA_DIR / "test.csv")

        track_dataframe(train_df, name="battery_train", context="train")
        track_dataframe(test_df, name="battery_test", context="test")

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_current_policy(ax=ax, hp=hp)
        mlflow.log_figure(fig, artifact_file="current_profile.png")

        res_for_plots = run_sims_from_tzero(rng=train_rng, hp=hp)

        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        sb.plot_rul_results(ax=axs[0], results=[res_for_plots])
        sb.plot_rul_bars(ax=axs[1], results=[res_for_plots])
        sb.plot_voltage_results(ax=axs[2], results=[res_for_plots])
        sb.plot_soc_results(ax=axs[3], results=[res_for_plots])

        mlflow.log_figure(fig, artifact_file="discharge_results.png")


if __name__ == "__main__":
    main()
