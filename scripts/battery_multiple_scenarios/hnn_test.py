from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import simbat as sb
from dotenv import load_dotenv
from flax import nnx

from qmodem.battery.data_generation import load_simulation_config
from qmodem.battery.evaluate import Hyperparameters, TestRULSamples, run_evaluation
from qmodem.battery.models import CNN, ConvType
from qmodem.tracking import get_run_parameters, retrieve_mlflow_setup_train
from qmodem.utils import setup_script_logging


def update_simulation_config(
    base_config: sb.SimulationConfig,
    idx: int,
    soc0: float,
    t0: float,
    tc_data: pd.DataFrame,
) -> sb.SimulationConfig:
    _to_update = ["soc_0", "t_0", "current_policies", "policy_choice_distribution"]

    def dirac_policy_choice_distribution() -> int:
        return 0

    # This is an important parameter. For the two policy case in this experiment,
    # it determines the time at which the policy is completely determined.
    t_event = 900

    # Get the policy at the given index
    policy_id = tc_data.loc[idx, "policy_id"]

    # Update to a deterministic policy if the event has happened
    if t0 > t_event:
        policy_choice_distribution = dirac_policy_choice_distribution
        current_policies = [base_config.current_policies[policy_id]]
    else:
        policy_choice_distribution = base_config.policy_choice_distribution
        current_policies = base_config.current_policies

    return sb.SimulationConfig(
        policy_choice_distribution=policy_choice_distribution,
        current_policies=current_policies,
        soc_0=soc0,
        t_0=t0,
        **{k: v for k, v in base_config.__dict__.items() if k not in _to_update},
    )


def reconstruct_true_rul_distribution(
    test_data: pd.DataFrame,
    simulator_base_config: sb.SimulationConfig,
    n_soc0s: int,
    n_mc_samples: int,
) -> TestRULSamples:
    # Count the number of unique test cases; create the output data structure
    n_tcs = test_data["run_id"].nunique()
    out = dict.fromkeys([f"test_case_{i}" for i in range(n_tcs)], None)

    # Loop over the test cases
    for tc_i in range(n_tcs):
        # Define an empty array to hold the RUL samples for this test case
        rul_samples = np.empty((n_soc0s, n_mc_samples), dtype=np.float32)

        # Extract the test case data and sort it by time
        tc_data = test_data[test_data["run_id"] == tc_i]
        tc_data = tc_data.sort_values(by="time", ascending=True).reset_index(drop=True)

        # Define the interemediate checkpoints and corresponding SOCs and times
        checkpoints = np.linspace(0, len(tc_data) - 1, num=n_soc0s, dtype=np.int32)
        soc0s = tc_data.loc[checkpoints, "soc"].to_numpy()
        t0s = tc_data.loc[checkpoints, "time"].to_numpy()

        # Loop over the checkpoints
        for i, (checkpoint, soc0, t0) in enumerate(zip(checkpoints, soc0s, t0s)):
            # Update the config with SOC, t_0, policies and policy choice distribution
            updated_config = update_simulation_config(
                base_config=simulator_base_config,
                idx=checkpoint,
                soc0=soc0,
                t0=t0,
                tc_data=tc_data,
            )

            # Simulate
            result = sb.simulate_constant_capacity_simple(
                n_sim=n_mc_samples, config=updated_config
            )

            # Update the checkpoint array
            rul_samples[i, :] = result.times_eod - t0

        # Update test case entry
        out[f"test_case_{tc_i}"] = rul_samples

    return out


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()

    hp = Hyperparameters()

    mlflow_setup = retrieve_mlflow_setup_train()
    run_parameters = get_run_parameters(mlflow_setup.run_id, mlflow_setup.backend_store)

    # Load the test raw data
    raw_data_dir = pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"])
    test_data = pd.read_csv(raw_data_dir / "test.csv")

    # Reconstruct the true RUL distribution for each test case
    simulator_base_config = load_simulation_config(
        run_id=os.environ["DATA_GEN_RUN_ID_MULTI"],
    )
    test_rul_samples = reconstruct_true_rul_distribution(
        test_data=test_data,
        simulator_base_config=simulator_base_config,
        n_soc0s=hp.test_n_soc0s,
        n_mc_samples=hp.test_n_mc_samples_simulator,
    )

    # Build a fresh model identical to the one used for training
    model = CNN(
        conv_type=ConvType.DETERMINISTIC,
        in_features=2,
        n_filters=int(run_parameters["conv_n_filters"]),
        kernel_size=int(run_parameters["conv_kernel_size"]),
        dropout_rate=float(run_parameters["dropout_rate"]),
        act_fn=getattr(nnx, run_parameters["activation_function"]),
        rngs=nnx.Rngs(0),
    )  # RNGs won't be used for inference, so the seed is arbitrary.

    # Main evaluation function
    run_evaluation(
        model=model,
        hp=hp,
        test_data=test_data,
        test_rul_samples=test_rul_samples,
        mlflow_setup=mlflow_setup,
        log_stream=log_stream,
        features=["load", "voltage"],
        train_mode=False,  # Deterministic network: aleatoric uncertainty only.
    )


if __name__ == "__main__":
    main()
