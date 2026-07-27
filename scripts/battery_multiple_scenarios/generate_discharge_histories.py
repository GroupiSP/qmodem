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
    bernoulli_policy_choice,
    gaussian_noise,
    log_simulation_config,
    write_histories,
)
from qmodem.battery.policies import VariableDischargeCurrentPolicy, plot_current_profile
from qmodem.tracking import MLFlowSetup, track_dataframe, track_mlflow

constant_cruise_policy = VariableDischargeCurrentPolicy(
    current_values=[-4.0, -1.0],
    time_values=[0.0, 600.0],
)
variable_cruise_policy = VariableDischargeCurrentPolicy(
    current_values=[-4.0, -1.0, -2.0, -1.0],
    time_values=[0.0, 600.0, 1800.0, 4000.0],
)


def make_simulator_config(
    rng: np.random.Generator,
    hp: Hyperparameters,
) -> sb.SimulationConfig:
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

    RAW_DATA_DIR = pathlib.Path(os.environ["RAW_DATA_DIR"])

    # MLFlow setup
    run_tags = {
        "case_study": "battery",
        "stage": "data_generation",
    }
    tracking_setup = MLFlowSetup(
        experiment_name="refactoring_jul_2026",
        run_name="generate_dummy_multiple",
        tags=run_tags,
    )

    with track_mlflow(tracking_setup):
        train_rng = np.random.default_rng(seed=hp.train_seed)
        test_rng = np.random.default_rng(seed=hp.test_seed)

        mlflow.log_params(dataclasses.asdict(hp))

        train_df = write_histories(
            make_simulator_config(train_rng, hp),
            n_histories=hp.n_histories_train + hp.n_histories_val,
        )
        test_config = make_simulator_config(test_rng, hp)
        test_df = write_histories(test_config, n_histories=hp.n_histories_test)

        # Log the test simulation config so it can be reloaded at evaluation time.
        log_simulation_config(test_config)

        train_df.to_csv(RAW_DATA_DIR / "train.csv", index=False)
        test_df.to_csv(RAW_DATA_DIR / "test.csv", index=False)

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
