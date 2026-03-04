"""Plot CRPS over time for Bayesian CNN (Flipout) RUL predictions.

At each evaluation time t along a stochastic discharge trajectory:
- The CNN receives the voltage window ending at t.
- N_WEIGHT_SAMPLES forward passes with different PRNG keys each yield (mu, var).
- One sample is drawn from N(mu, sqrt(var)) per pass, giving N_WEIGHT_SAMPLES samples
  that capture both epistemic (weight) and aleatoric (output Gaussian) uncertainty.
- Stochastic simulations from SoC(t) produce the reference RUL distribution.
- CRPS measures the distance between the two distributions.

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

from qmodem import BayesCNN1D, FlipoutConv1D  # noqa: E402
from qmodem.metrics import crps  # noqa: E402


def main() -> None:
    np.random.seed(TEST_SEED)

    # Configuration
    N_SIMU = 500  # Stochastic simulations per eval point (reference distribution)
    N_WEIGHT_SAMPLES = 500  # Bayesian forward passes (predicted distribution)
    N_EVAL_POINTS = 50  # Number of evaluation time points along the trajectory

    # Directories
    root_dir, _, METADATA_DIR = get_run_dirs("bayes_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"
    output_dir = Path("saved/bayes_cnn/crps_over_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(METADATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    window_size = metadata["training_params"]["window_size"]
    y_max_train = metadata["scaling_params"]["y_max"]
    dt = metadata["simulator_config"]["dt"]

    print("=" * 70)
    print("CRPS Over Time — Bayesian CNN (Flipout)")
    print("=" * 70)
    print(f"Stochastic sims per eval point: {N_SIMU}")
    print(f"Bayesian weight samples: {N_WEIGHT_SAMPLES}")
    print(f"Evaluation points: {N_EVAL_POINTS}")
    print()

    # Load trained model
    print("Loading trained Bayesian CNN model...")
    model = restore_model_from_checkpoint(
        ckpt_dir / "trained_state",
        lambda: BayesCNN1D(
            conv_cls=FlipoutConv1D, **metadata["model_params"], rngs=nnx.Rngs(0)
        ),
    )
    print("Model loaded successfully.")
    print()

    # Run a single stochastic simulation as the observed trajectory.
    print("Running stochastic test simulation...")
    sim_config = metadata["simulator_config"].copy()
    sim_config["N_simu"] = 1
    sim_0 = les.SimulatorSimple(sim_config)
    sim_0.simulate()

    discharge_voltage = sim_0.v_memo.flatten()  # shape (N_t,)
    socs = sim_0.soc_memo.flatten()  # shape (N_t,)
    N_t = len(discharge_voltage)

    print(f"Trajectory length: {N_t} steps")
    print()

    # Select evaluation time indices (must have a full window available).
    first_valid = window_size
    eval_indices = np.linspace(first_valid, N_t - 1, N_EVAL_POINTS, dtype=int)

    base_key = jax.random.PRNGKey(42)
    ts_eval = []
    crps_values = []

    print("Computing CRPS at each evaluation point...")
    for k, idx in enumerate(eval_indices):
        t = idx * dt
        ts_eval.append(t)

        # --- Predicted distribution (Bayesian CNN) ---
        start = idx - window_size
        window = discharge_voltage[start:idx].reshape(1, -1)
        x_input = jnp.expand_dims(window, 0)

        # Bayesian sampling: N_WEIGHT_SAMPLES forward passes, one sample per pass
        pred_samples = []
        for i in range(N_WEIGHT_SAMPLES):
            key = jax.random.fold_in(base_key, k * N_WEIGHT_SAMPLES + i)
            pred = model(x_input, rngs=nnx.Rngs(params=key))[0]  # shape (2,)
            mu = float(pred[0]) * y_max_train
            var = float(pred[1]) * y_max_train**2
            std = np.sqrt(max(var, 1e-12))
            sample = np.clip(np.random.normal(mu, std), 0, None)
            pred_samples.append(sample)

        pred_samples = np.array(pred_samples)

        # --- Reference distribution (stochastic simulations from SoC at t) ---
        ref_config = metadata["simulator_config"].copy()
        ref_config["SoC_0"] = float(socs[idx])
        ref_config["N_simu"] = N_SIMU
        ref_sim = les.SimulatorSimple(ref_config)
        ref_sim.simulate()
        ref_samples = np.array(ref_sim.t_eods)

        # --- CRPS ---
        all_samples = np.concatenate([pred_samples, ref_samples])
        x_grid = jnp.linspace(0, float(all_samples.max()) * 1.1, 500)
        crps_val = float(crps(jnp.array(pred_samples), jnp.array(ref_samples), x_grid))
        crps_values.append(crps_val)

        print(
            f"  [{k + 1:2d}/{N_EVAL_POINTS}] t={t:7.1f}s | "
            f"mu={np.mean(pred_samples):7.1f} | std={np.std(pred_samples):6.1f} | "
            f"CRPS={crps_val:.3f}"
        )

    print()

    # Plot CRPS over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts_eval, crps_values, marker="o", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CRPS [s]")
    ax.set_title("CRPS Over Time — Bayesian CNN (Flipout) vs Simulator")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "crps_over_time.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_dir / 'crps_over_time.png'}")
    plt.show()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
