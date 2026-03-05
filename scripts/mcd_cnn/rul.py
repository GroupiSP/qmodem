"""Compare MC Dropout CNN predictions with simulator predictions for battery RUL.

This script:
- Loads a pre-generated test case (discharge trajectory + reference RUL distributions)
- Uses trained MCDCNN1D to predict RUL with uncertainty (MC Dropout)
- At each time window, runs N_MC_PASSES forward passes in train mode;
  each pass yields (mu, var), and one sample is drawn from N(mu, sqrt(var))
- Compares predictions using 95% CI plots

Requires a trained MCDCNN1D checkpoint from ``train.py``
and pre-generated test data from ``scripts/generate_data.py``.
"""

import pickle
import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import TEST_SEED  # noqa: E402
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
    N_MC_PASSES = 500  # Number of forward passes to reconstruct the model distribution

    print("=" * 70)
    print("MC Dropout CNN RUL Prediction with Uncertainty")
    print("=" * 70)
    print(f"Number of MC forward passes: {N_MC_PASSES}")
    print()

    # Load pre-generated test case
    print("Loading test case data...")
    test_data = np.load("data/test_case_0.npz")
    discharge_voltage = test_data["voltage"]
    t_eod = float(test_data["t_eod"])
    dt = float(test_data["dt"])
    eval_indices = test_data["eval_indices"]
    ref_t_eods = test_data["ref_t_eods"]
    N_t = len(discharge_voltage)
    N_INTERMEDIATE_SOCs = len(eval_indices)
    print(f"Trajectory length: {N_t} steps, t_eod={t_eod:.1f}s")
    print()

    # Load trained model
    print("Loading trained MC Dropout CNN model...")
    model = MCDCNN1D(**metadata["model_params"], rngs=nnx.Rngs(params=0, dropout=1))
    restore_model_state(ckpt_dir / "trained_state", model)
    print("Model loaded successfully")
    print()

    # Time-window the voltage trajectory
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

    # Cover the last part of the trajectory if it doesn't fit a full window
    if end < N_t:
        ts_pred.append(N_t * dt)
        X = discharge_voltage[-window_size:].reshape(1, -1)
        x_input = jnp.expand_dims(X, 0)

        mu_samples = []
        full_samples = []
        for _ in range(N_MC_PASSES):
            pred = model(x_input, rngs=rng_dropout)[0]
            mu = float(pred[0]) * y_max_train
            var = float(pred[1]) * y_max_train**2
            std = np.sqrt(max(var, 1e-12))
            mu_samples.append(mu)
            sample = np.clip(np.random.normal(mu, std), 0, None)
            full_samples.append(sample)

        pred_means.append(np.mean(mu_samples))
        pred_lowers.append(np.percentile(full_samples, 2.5))
        pred_uppers.append(np.percentile(full_samples, 97.5))

    print("Computing reference RUL confidence intervals from pre-generated data...")
    ruls_true_lowers = []
    ruls_true_uppers = []
    for k in range(N_INTERMEDIATE_SOCs):
        t_eods = ref_t_eods[k]
        ruls_true_lowers.append(np.percentile(t_eods, 2.5))
        ruls_true_uppers.append(np.percentile(t_eods, 97.5))

    # true RUL is linear
    ts_rul_true = np.linspace(0.0, N_t, N_INTERMEDIATE_SOCs) * dt
    ruls_true = t_eod - ts_rul_true

    print("Part 3. Plotting results...")
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    ax0.plot(
        ts_rul_true,
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
