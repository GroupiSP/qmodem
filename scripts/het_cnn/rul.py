"""Compare heteroscedastic CNN predictions with simulator predictions for battery RUL.

This script:
- Runs a deterministic simulation to get voltage trajectory
- Runs stochastic simulations from SOCs after the first time window
- Uses trained HeteroscedasticCNN1DV1 to predict RUL with uncertainty
- Compares predictions using 95% CI plots and CRPS metric
- Plots CDFs of simulator and CNN predictions
"""

import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _seeds import TEST_SEED  # noqa: E402
from _shared import (  # noqa: E402
    get_run_dirs,
    restore_model_from_checkpoint,
)

from qmodem import HeteroscedasticCNN1D


def main() -> None:
    np.random.seed(TEST_SEED)

    # Directories
    root_dir, _, METADATA_DIR = get_run_dirs("het_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"

    # Create output directory for plots
    output_dir = Path("saved/het_cnn/rul")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(METADATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Battery simulator parameters
    N_SIMU = 100  # Number of stochastic simulations to reconstruct the true RUL distributions

    print("=" * 70)
    print("Heteroscedastic CNN RUL Prediction with Uncertainty")
    print("=" * 70)
    print(f"Number of stochastic simulations: {N_SIMU}")
    print()

    # Recreate the simulator used in training and run a single simulation.
    print("Part 1. Single simulation to get voltage and SoC history...")
    sim_0 = les.SimulatorSimple(metadata["simulator_config"])
    sim_0.simulate()

    dt = metadata["simulator_config"]["dt"]
    N_t = sim_0.v_memo.shape[0]

    # Load trained model
    print("Loading trained heteroscedastic CNN model...")
    model = restore_model_from_checkpoint(
        ckpt_dir / "trained_state",
        lambda: HeteroscedasticCNN1D(**metadata["model_params"], rngs=nnx.Rngs(0)),
    )
    print("Model loaded successfully")
    print()

    # Time-window the voltage trajectory
    discharge_voltage = sim_0.v_memo.flatten()
    window_size = metadata["training_params"]["window_size"]
    stride = metadata["training_params"]["stride"]
    y_max_train = metadata["scaling_params"]["y_max"]
    model.eval()

    ts_pred = []
    pred_means = []
    pred_vars = []
    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        ts_pred.append(end * dt)  # Time corresponding to the end of the window
        X = discharge_voltage[start:end].reshape(1, -1)
        pred = model(jnp.expand_dims(X, 0))[0]  # Shape: (2,)
        pred_means.append(
            pred[0] * y_max_train
        )  # Mean is the zero-th element of the output
        pred_vars.append(
            pred[1] * y_max_train**2
        )  # Variance is the first element, scaled by y_max^2

    print(
        "Part 2. Running stochastic simulations from intermediate SOCs and comparing with CNN predictions..."
    )
    # Get 10 intermediate SoCs from the previous simulation, run stochastic simulations from each and save
    # the RUL distributions.
    ruls_true = []
    ruls_true_lowers = []
    ruls_true_uppers = []
    socs = sim_0.soc_memo.flatten()
    for i in range(0, N_t, N_t // 10):
        soc_0 = socs[i]
        sim_config = metadata["simulator_config"].copy()
        sim_config["SoC_0"] = soc_0
        sim_config["N_simu"] = N_SIMU
        sim = les.SimulatorSimple(sim_config)
        sim.simulate()
        t_eods = sim.t_eods
        ruls_true.append(
            np.mean(t_eods)
        )  # Use the expected RUL from the simulator as the "true" RUL at this point
        # Calculate 95% confidence intervals for the true RUL distribution using the variance from the simulator
        ruls_true_lowers.append(np.percentile(t_eods, 2.5))
        ruls_true_uppers.append(np.percentile(t_eods, 97.5))

    print("Part 3. Plotting results...")

    fig0, ax0 = plt.subplots(figsize=(10, 6))
    ax0.plot(
        np.arange(0, N_t, N_t // 10) * dt,
        ruls_true,
        label="True RUL",
    )
    ax0.plot(
        ts_pred, pred_means, label="Predicted RUL (Heteroscedastic CNN)", marker="o"
    )
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("RUL [s]")
    ax0.set_title("Heteroscedastic CNN RUL Mean Predictions")
    ax0.set_ylim(bottom=0.0)
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    fig0.savefig(output_dir / "rul_point_prediction.png", dpi=150, bbox_inches="tight")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    ax1.plot(
        np.arange(0, N_t, N_t // 10) * dt, ruls_true, label="True RUL", color=colors[0]
    )
    ax1.fill_between(
        np.arange(0, N_t, N_t // 10) * dt,
        ruls_true_lowers,
        ruls_true_uppers,
        color=colors[0],
        alpha=0.2,
        label="True RUL 95% CI",
    )
    ax1.plot(
        ts_pred,
        pred_means,
        label="Predicted RUL (Heteroscedastic CNN)",
        color=colors[1],
        marker="o",
    )
    ax1.fill_between(
        ts_pred,
        np.array(pred_means) - 1.96 * np.sqrt(pred_vars),
        np.array(pred_means) + 1.96 * np.sqrt(pred_vars),
        color=colors[1],
        alpha=0.2,
        label="Predicted RUL 95% CI",
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("RUL [s]")
    ax1.set_title("Heteroscedastic CNN RUL Predictions with Uncertainty")
    ax1.set_ylim(bottom=0.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig1.savefig(
        output_dir / "rul_uncertainty_prediction.png", dpi=150, bbox_inches="tight"
    )
    plt.show()
    print()
    print("Done!")


if __name__ == "__main__":
    main()
