"""Compare Bayesian CNN (Flipout) predictions with simulator predictions for battery
RUL.

This script:
- Runs a deterministic simulation to get voltage trajectory
- Runs stochastic simulations from SOCs after the first time window
- Uses trained BayesCNN1D to predict RUL with uncertainty
- Point prediction: averages μ_out across M weight samples (Bayesian predictive mean)
- Uncertainty: draws one sample from N(μ_out, σ²_out) per weight sample to capture
  both epistemic and aleatoric uncertainty
- Compares predictions using 95% CI plots

Requires a trained BayesCNN1D checkpoint from ``train.py``.
"""

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import TEST_SEED  # noqa: E402
from _shared import (  # noqa: E402
    get_run_dirs,
    restore_model_from_checkpoint,
)

from qmodem import BayesCNN1D, FlipoutConv1D


def main() -> None:
    np.random.seed(TEST_SEED)

    # Directories
    root_dir, _, METADATA_DIR = get_run_dirs("bayes_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"

    # Create output directory for plots
    output_dir = Path("saved/bayes_cnn/rul")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(METADATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # Bayesian sampling parameters
    N_WEIGHT_SAMPLES = 500  # Forward passes for uncertainty quantification
    N_SIMU = 500  # Stochastic simulations for reference RUL distributions
    N_INTERMEDIATE_SOCs = 200

    print("=" * 70)
    print("Bayesian CNN (Flipout) RUL Prediction with Uncertainty")
    print("=" * 70)
    print(f"Number of weight samples: {N_WEIGHT_SAMPLES}")
    print(f"Number of stochastic simulations: {N_SIMU}")
    print()

    # Recreate the simulator used in training and run a single simulation.
    print("Part 1. Single simulation to get voltage and SoC history...")
    sim_0 = les.SimulatorSimple(metadata["simulator_config"])
    sim_0.simulate()

    dt = metadata["simulator_config"]["dt"]
    N_t = sim_0.v_memo.shape[0]

    # Load trained model
    print("Loading trained Bayesian CNN model...")
    model = restore_model_from_checkpoint(
        ckpt_dir / "trained_state",
        lambda: BayesCNN1D(
            conv_cls=FlipoutConv1D, **metadata["model_params"], rngs=nnx.Rngs(0)
        ),
    )
    print("Model loaded successfully")
    print()

    # Time-window the voltage trajectory
    discharge_voltage = sim_0.v_memo.flatten()
    window_size = metadata["training_params"]["window_size"]
    stride = metadata["training_params"]["stride"]
    y_max_train = metadata["scaling_params"]["y_max"]

    base_key = jax.random.PRNGKey(42)

    ts_pred = []
    pred_means = []  # Bayesian predictive mean: avg of μ_out
    pred_lowers = []  # 2.5th percentile of full predictive samples
    pred_uppers = []  # 97.5th percentile of full predictive samples
    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        ts_pred.append(end * dt)
        X = discharge_voltage[start:end].reshape(1, -1)
        x_input = jnp.expand_dims(X, 0)

        # Collect μ_out for point prediction and full samples for uncertainty
        mu_samples = []
        full_samples = []
        for i in range(N_WEIGHT_SAMPLES):
            key = jax.random.fold_in(base_key, i)
            pred = model(x_input, rngs=nnx.Rngs(params=key))[0]  # Shape: (2,)
            mu = float(pred[0]) * y_max_train
            var = float(pred[1]) * y_max_train**2
            std = np.sqrt(max(var, 1e-12))
            mu_samples.append(mu)
            sample = np.clip(np.random.normal(mu, std), 0, None)
            full_samples.append(sample)

        # Point prediction: average of μ_out (Bayesian predictive mean)
        pred_means.append(np.mean(mu_samples))
        # Uncertainty: 95% confidence intervals of full predictive samples
        pred_lowers.append(np.percentile(full_samples, 2.5))
        pred_uppers.append(np.percentile(full_samples, 97.5))

    print(
        "Part 2. Running stochastic simulations from intermediate SOCs and comparing "
        "with Bayesian CNN predictions..."
    )
    # Get intermediate SoCs from the previous simulation
    ruls_true_lowers = []
    ruls_true_uppers = []
    socs = sim_0.soc_memo.flatten()
    for i in np.linspace(0, N_t, N_INTERMEDIATE_SOCs, endpoint=False, dtype=np.int32):
        soc_0 = socs[i]
        sim_config = metadata["simulator_config"].copy()
        sim_config["SoC_0"] = soc_0
        sim_config["N_simu"] = N_SIMU
        sim = les.SimulatorSimple(sim_config)
        sim.simulate()
        t_eods = sim.t_eods
        # Calculate 95% confidence intervals for the true RUL distribution using the variance from the simulator
        ruls_true_lowers.append(np.percentile(t_eods, 2.5))
        ruls_true_uppers.append(np.percentile(t_eods, 97.5))

    # true RUL is linear
    ts_rul_true = np.linspace(0.0, N_t, N_INTERMEDIATE_SOCs) * dt
    ruls_true = sim_0.t_eods[0] - ts_rul_true

    print("Part 3. Plotting results...")
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    ax0.plot(ts_rul_true, ruls_true, label="True RUL")
    ax0.plot(
        ts_pred, pred_means, label="Predicted RUL (Bayesian CNN Flipout)", marker="o"
    )
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("RUL [s]")
    ax0.set_title("Bayesian CNN (Flipout) RUL Mean Predictions")
    ax0.set_ylim(bottom=0.0)
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    fig0.savefig(output_dir / "rul_point_prediction.png", dpi=150, bbox_inches="tight")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    ax1.plot(
        ts_rul_true,
        ruls_true,
        label="True RUL",
        color=colors[0],
    )
    ax1.fill_between(
        ts_rul_true,
        ruls_true_lowers,
        ruls_true_uppers,
        color=colors[0],
        alpha=0.2,
        label="True RUL 95% CI",
    )
    ax1.plot(
        ts_pred,
        pred_means,
        label="Predicted RUL (Bayesian CNN Flipout)",
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
    ax1.set_title("Bayesian CNN (Flipout) RUL Predictions with Uncertainty")
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
