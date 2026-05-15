from __future__ import annotations

import pathlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto

import lib_eod_simulation as les
import mlflow
import numpy as np
import pandas as pd

from qmodem.tracking import MLFlowSetup, track_mlflow

# TODO: track the data generator parameters (and the data itself?) with MLFlow.


class ProcessNoiseDistribution(StrEnum):
    NORMAL = auto()
    UNIFORM = auto()
    ZERO = auto()


class CurrentPolicy(StrEnum):
    CONSTANT = auto()


class VOCModel(StrEnum):
    BUSTOS_BAEZA = auto()


class ECMModel(StrEnum):
    THEVENIN_ZERO_ORDER = auto()


dist_name_to_params = {
    ProcessNoiseDistribution.NORMAL: lambda loc, scale: {"loc": loc, "scale": scale},
    ProcessNoiseDistribution.UNIFORM: lambda low, high: {"low": low, "high": high},
    ProcessNoiseDistribution.ZERO: lambda: {},
}

policy_name_to_params = {
    CurrentPolicy.CONSTANT: lambda amplitude: {"amplitude": amplitude},
}

ecm_model_name_to_params = {
    ECMModel.THEVENIN_ZERO_ORDER: lambda r0: {"r0": r0},
}


@dataclass(frozen=True)
class Hyperparameters:
    current_policy: CurrentPolicy = CurrentPolicy.CONSTANT
    current_policy_params: dict[str, float] = field(
        default_factory=lambda: policy_name_to_params[CurrentPolicy.CONSTANT](
            amplitude=-2.8 * 0.75
        )
    )  # in Amperes, negative for discharge
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
    process_noise_params: dict[str, float] = field(
        default_factory=lambda: dist_name_to_params[ProcessNoiseDistribution.NORMAL](
            loc=0.0, scale=3e-3
        )
    )
    measurement_noise_params: dict[str, float] = field(
        default_factory=lambda: dist_name_to_params[ProcessNoiseDistribution.ZERO]()
    )
    soc_range_train_val: tuple[float, float] = (0.05, 1.0)
    train_seed: int = 42
    test_seed: int = 123


hp = Hyperparameters()


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


def generate_train(rng: np.random.Generator) -> pd.DataFrame:
    soc_0s = rng.uniform(
        low=hp.soc_range_train_val[0],
        high=hp.soc_range_train_val[1],
        size=hp.n_histories_train + hp.n_histories_val,
    )

    out_df = pd.DataFrame(columns=["run_id", "time", "soc", "voltage"])

    for i, soc_0 in enumerate(soc_0s):
        config = les.SimulationConfig(
            process_noise_distribution=lambda: rng.normal(**hp.process_noise_params),
            measurement_noise_distribution=lambda: 0.0,
            dt=hp.dt,
            soc_0=soc_0,
        )
        result = les.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()

        # Modify the dataframe and append it to the output one
        _modify_dataframe(df, i)

        out_df = pd.concat([out_df, df], ignore_index=True)

    return out_df


def generate_test(rng: np.random.Generator) -> pd.DataFrame:
    out_df = pd.DataFrame(columns=["run_id", "time", "soc", "voltage"])

    for i in range(hp.n_histories_test):
        config = les.SimulationConfig(
            process_noise_distribution=lambda: rng.normal(**hp.process_noise_params),
            measurement_noise_distribution=lambda: 0.0,
            dt=hp.dt,
            soc_0=1.0,
        )
        result = les.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()

        _modify_dataframe(df, i)

        out_df = pd.concat([out_df, df], ignore_index=True)

    return out_df


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

    # MLFlow setup
    run_tags = {
        "case_study": "battery",
        "stage": "data_generation",
    }
    tracking_setup = MLFlowSetup(
        experiment_name="battery_default",
        run_name="generate_discharge_histories",
        tags=run_tags,
    )
    with track_mlflow(tracking_setup):
        train_rng = np.random.default_rng(seed=hp.train_seed)
        test_rng = np.random.default_rng(seed=hp.test_seed)

        mlflow.log_params(asdict(hp))

        train_df = generate_train(train_rng)
        test_df = generate_test(test_rng)

        save_dataframe_to_file(train_df, path=BATTERY_DATA_DIR / "train.csv")
        save_dataframe_to_file(test_df, path=BATTERY_DATA_DIR / "test.csv")

        track_dataframe(train_df, name="battery_train", context="train")
        track_dataframe(test_df, name="battery_test", context="test")


if __name__ == "__main__":
    main()
