"""Compare MC Dropout CNN predictions with simulator predictions for battery RUL.

This script:
- Runs a deterministic simulation to get voltage trajectory
- Runs stochastic simulations from SOCs after the first time window
- Uses trained MCDCNN1D to predict RUL with uncertainty (MC Dropout)
- At each time window, runs N_MC_PASSES forward passes in train mode;
  each pass yields (mu, var), and one sample is drawn from N(mu, sqrt(var))
- Compares predictions using 95% CI plots

Requires a trained MCDCNN1D checkpoint from ``train.py``.
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
    restore_model_state,
)

from qmodem import MCDCNN1D


def main() -> None:
    np.random.seed(TEST_SEED)

    # Directories
    root_dir, _, METADATA_DIR = get_run_dirs("mcd_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"

    # Create output directory for plots
    output_dir = Path("saved/mcd_cnn/rul")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(METADATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # MC Dropout parameters
    N_MC_PASSES = 100  # Number of forward passes to reconstruct the model distribution
    N_SIMU = 100  # Number of stochastic simulations for reference RUL distributions

    print("=" * 70)
    print("MC Dropout CNN RUL Prediction with Uncertainty")
    print("=" * 70)
    print(f"Number of MC forward passes: {N_MC_PASSES}")
    print(f"Number of stochastic simulations: {N_SIMU}")
    print()

    # Recreate the simulator used in training and run a single simulation.
    print("Part 1. Single simulation to get voltage and SoC history...")
    sim_0 = les.SimulatorSimple(metadata["simulator_config"])
    sim_0.simulate()

    dt = metadata["simulator_config"]["dt"]
    N_t = sim_0.v_memo.shape[0]

    # Load trained model
    print("Loading trained MC Dropout CNN model...")
    model = MCDCNN1D(**metadata["model_params"], rngs=nnx.Rngs(params=0, dropout=1))
    restore_model_state(ckpt_dir / "trained_state", model)
    print("Model loaded successfully")
    print()

    # Time-window the voltage trajectory
    discharge_voltage = sim_0.v_memo.flatten()
    window_size = metadata["training_params"]["window_size"]
    stride = metadata["training_params"]["stride"]
    y_max_train = metadata["scaling_params"]["y_max"]

    # MC Dropout: model in train mode for stochastic forward passes
    model.train()
    rng_dropout = nnx.Rngs(dropout=42)

    ts_pred = []
    pred_means = []
    pred_lowers = []
    pred_uppers = []
    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        ts_pred.append(end * dt)
        X = discharge_voltage[start:end].reshape(1, -1)
        x_input = jnp.expand_dims(X, 0)

        # MC sampling: N forward passes, collect μ_out and full samples separately
        mu_samples = []
        full_samples = []
        for _ in range(N_MC_PASSES):
            pred = model(x_input, rngs=rng_dropout)[0]  # Shape: (2,)
            mu = float(pred[0]) * y_max_train
            var = float(pred[1]) * y_max_train**2
            std = np.sqrt(max(var, 1e-12))
            mu_samples.append(mu)
            sample = np.clip(np.random.normal(mu, std), 0, None)
            full_samples.append(sample)

        # Point prediction: average of μ_out (epistemic uncertainty only)
        pred_means.append(np.mean(mu_samples))
        # Uncertainty: 95% confidence intervals of full predictive samples (epistemic + aleatoric)
        pred_lowers.append(np.percentile(full_samples, 2.5))
        pred_uppers.append(np.percentile(full_samples, 97.5))

    print(
        "Part 2. Running stochastic simulations from intermediate SOCs and comparing with CNN predictions..."
    )
    # Get 10 intermediate SoCs from the previous simulation
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
    ax0.plot(ts_pred, pred_means, label="Predicted RUL (MC Dropout CNN)", marker="o")
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("RUL [s]")
    ax0.set_title("MC Dropout CNN RUL Mean Predictions")
    ax0.set_ylim(bottom=0.0)
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    fig0.savefig(output_dir / "rul_point_prediction.png", dpi=150, bbox_inches="tight")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    ax1.plot(
        np.arange(0, N_t, N_t // 10) * dt,
        ruls_true,
        label="True RUL",
        color=colors[0],
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
        label="Predicted RUL (MC Dropout CNN)",
        color=colors[1],
        marker="o",
    )
    ax1.fill_between(
        ts_pred,
        pred_lowers,
        pred_uppers,
        color=colors[1],
        alpha=0.2,
        label="Predicted RUL 95% CI",
    )
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("RUL [s]")
    ax1.set_title("MC Dropout CNN RUL Predictions with Uncertainty")
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
