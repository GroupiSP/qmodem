from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum, auto

import lib_eod_simulation as les
import mlflow
import numpy as np

from qmodem.tracking import MLFlowSetup, track_mlflow
from qmodem.utils import BATTERY_DATA_DIR_PATH

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


def generate_train(rng: np.random.Generator) -> None:
    soc_0s = rng.uniform(
        low=hp.soc_range_train_val[0],
        high=hp.soc_range_train_val[1],
        size=hp.n_histories_train + hp.n_histories_val,
    )

    for i, soc_0 in enumerate(soc_0s):
        config = les.SimulationConfig(
            process_noise_distribution=lambda: rng.normal(**hp.process_noise_params),
            measurement_noise_distribution=lambda: 0.0,
            dt=hp.dt,
            soc_0=soc_0,
        )
        result = les.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()
        df.to_csv(BATTERY_DATA_DIR_PATH / f"train_history_{i}.csv", index=False)

    return


def generate_test(rng: np.random.Generator) -> None:
    for i in range(hp.n_histories_test):
        config = les.SimulationConfig(
            process_noise_distribution=lambda: rng.normal(**hp.process_noise_params),
            measurement_noise_distribution=lambda: 0.0,
            dt=hp.dt,
            soc_0=1.0,
        )
        result = les.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()
        df.to_csv(BATTERY_DATA_DIR_PATH / f"test_history_{i}.csv", index=False)

    return


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

        generate_train(train_rng)
        generate_test(test_rng)


if __name__ == "__main__":
    main()
