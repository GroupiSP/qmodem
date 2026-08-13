from __future__ import annotations

import os
import pathlib
from dataclasses import asdict

import jax
import mlflow
import pandas as pd
from dotenv import load_dotenv

from qmodem.battery.data_generation import (
    load_simulation_config,
    reconstruct_true_rul_distribution,
    sim_updater_two_scenarios,
)
from qmodem.battery.dispatch import ModelBuildParameters, build_model
from qmodem.battery.evaluate import (
    TestHyperparameters,
    log_evaluation_metrics,
    restore_model_state,
    run_evaluation,
)
from qmodem.battery.tracking import mlflow_load_scaler
from qmodem.tracking import (
    get_run_parameters,
    retrieve_mlflow_setup_train,
    track_mlflow,
)
from qmodem.utils import setup_script_logging


def main() -> None:
    load_dotenv(override=True)

    log_stream = setup_script_logging()

    hp = TestHyperparameters()

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
        simulator_updater=sim_updater_two_scenarios,
        event_fn=lambda t0, soc0: t0 > 900.0,
    )

    # Build a fresh model identical to the one used for training
    model = build_model(ModelBuildParameters(**run_parameters))

    with track_mlflow(setup=mlflow_setup) as run:
        # Load the scalers fitted on the training data.
        x_scaler = mlflow_load_scaler(f"runs:/{run.info.run_id}/x_scaler")
        y_scaler = mlflow_load_scaler(f"runs:/{run.info.run_id}/y_scaler")

        restore_model_state(model, run.info.run_id)

        model.eval()  # NOTE MCD is overwritten to redirect the eval mode to train mode for stochastic predictions

        window_size = int(run.data.params["window_size"])

        base_key = jax.random.key(hp.test_rng_seed)
        results = run_evaluation(
            model=model,
            test_data=test_data,
            test_rul_samples=test_rul_samples,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            window_size=window_size,
            n_soc0s=hp.test_n_soc0s,
            n_mc_samples=hp.test_n_mc_samples_model,
            features=["load", "voltage"],
            key=base_key,
        )

        # Log parameters and metrics with MLflow.
        mlflow.log_params(asdict(hp))
        log_evaluation_metrics(results, hp)
        mlflow.log_text(log_stream.getvalue(), artifact_file="test_log.txt")


if __name__ == "__main__":
    main()
