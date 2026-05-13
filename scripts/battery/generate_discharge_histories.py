from __future__ import annotations

import lib_eod_simulation as les
import numpy as np

from qmodem.utils import BATTERY_DATA_DIR_PATH


def generate_train(rng: np.random.Generator, n_histories: int) -> None:
    soc_0s = rng.uniform(low=0.05, high=1.0, size=n_histories)

    for i, soc_0 in enumerate(soc_0s):
        config = les.SimulationConfig(
            process_noise_distribution=lambda: rng.normal(loc=0.0, scale=3e-3),
            measurement_noise_distribution=lambda: 0.0,
            dt=20.0,
            soc_0=soc_0,
        )
        result = les.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()
        df.to_csv(BATTERY_DATA_DIR_PATH / f"train_history_{i}.csv", index=False)

    return


def generate_test(rng: np.random.Generator, n_histories: int) -> None:
    config = les.SimulationConfig(
        process_noise_distribution=lambda: rng.normal(loc=0.0, scale=3e-3),
        measurement_noise_distribution=lambda: 0.0,
        dt=20.0,
        soc_0=1.0,
    )
    result = les.simulate_constant_capacity_simple(n_sim=n_histories, config=config)
    df = result.to_dataframe()
    df.to_csv(BATTERY_DATA_DIR_PATH / "test_histories.csv", index=False)

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
    train_rng = np.random.default_rng(seed=42)
    test_rng = np.random.default_rng(seed=123)

    generate_train(train_rng, n_histories=120)
    generate_test(test_rng, n_histories=10)


if __name__ == "__main__":
    main()
