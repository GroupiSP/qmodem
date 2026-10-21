from __future__ import annotations

import dataclasses
import functools
import os
import pathlib

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import simbat as sb
from dotenv import load_dotenv

from qmodem.battery.data_generation import (
    Hyperparameters,
    add_rul_series,
    bernoulli_policy_choice,
    gaussian_noise,
    log_simulation_config,
    write_split_histories,
)
from qmodem.battery.policies import VariableDischargeCurrentPolicy, plot_current_profile
from qmodem.tracking import MLFlowSetup, track_dataframe, track_mlflow

constant_cruise_policy = VariableDischargeCurrentPolicy(
    current_values=[-5.0, -2.5],
    time_values=[0.0, 100.0],
)
variable_cruise_policy = VariableDischargeCurrentPolicy(
    current_values=[-5.0, -2.5, -4.0, -2.5],
    time_values=[0.0, 100.0, 900.0, 2600.0],
)


def make_simulator_config(
    rng: np.random.Generator,
    hp: Hyperparameters,
) -> sb.SimulationConfig:
    """Build the two-policy simulation config whose random draws come from ``rng``.

    Args:
        rng: Generator of the policy choice and of the process and measurement noise.
        hp: Data-generation hyperparameters.

    Returns:
        The simulation config.
    """
    return sb.SimulationConfig(
        current_policies=[constant_cruise_policy, variable_cruise_policy],
        policy_choice_distribution=functools.partial(
            bernoulli_policy_choice, rng=rng, p=(0.7, 0.3)
        ),
        process_noise_distribution=functools.partial(
            gaussian_noise,
            rng=rng,
            noise_std=hp.process_noise_std,
        ),
        measurement_noise_distribution=functools.partial(
            gaussian_noise,
            rng=rng,
            noise_std=hp.measurement_noise_std,
        ),
        dt=hp.dt,
        soc_0=1.0,
    )


def main() -> None:
    load_dotenv()

    hp = Hyperparameters()

    RAW_DATA_DIR = pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"])
    existing = [RAW_DATA_DIR / f for f in ("train.csv", "test.csv")]
    if any(f.exists() for f in existing):
        raise FileExistsError(
            f"{RAW_DATA_DIR} already holds generated data; point RAW_DATA_DIR_MULTI "
            "to a new directory instead of overwriting it."
        )
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # MLFlow setup
    tracking_setup = MLFlowSetup(run_name="generate_histories")

    with track_mlflow(tracking_setup):
        mlflow.log_params(dataclasses.asdict(hp))

        make_config = functools.partial(make_simulator_config, hp=hp)
        # Train, backbone validation set and posterior validation set: contiguous
        # run_id blocks of train.csv, each with fixed policy counts and its own
        # random stream spawned from train_seed.
        train_df = write_split_histories(
            make_config, hp.train_stream_splits(), hp.train_seed
        )
        test_df = write_split_histories(make_config, [hp.test_split()], hp.test_seed)
        # The pickled test config keeps the 70/30 policy prior only for the
        # evaluation reference band; the test histories use fixed policy counts.
        test_config = make_simulator_config(np.random.default_rng(hp.test_seed), hp)

        train_df = add_rul_series(train_df)
        test_df = add_rul_series(test_df)

        # Log the test simulation config so it can be reloaded at evaluation time.
        log_simulation_config(test_config)

        train_df.to_csv(RAW_DATA_DIR / "train.csv", index=False)
        test_df.to_csv(RAW_DATA_DIR / "test.csv", index=False)

        track_dataframe(train_df, name="battery_train", context="train")
        track_dataframe(test_df, name="battery_test", context="test")

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        plot_current_profile(
            ax=axs[0], policy=constant_cruise_policy, t_grid=np.linspace(0, 4000, 100)
        )
        axs[0].set_title("Constant Cruise Policy")

        plot_current_profile(
            ax=axs[1], policy=variable_cruise_policy, t_grid=np.linspace(0, 4000, 100)
        )
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
