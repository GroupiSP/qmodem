"""Plot CRPS over time for heteroscedastic CNN RUL predictions.

At each evaluation time t along a stochastic discharge trajectory:
- The CNN receives the voltage window ending at t and outputs (mu, var).
- Samples from N(mu, var), clipped at zero, form the predicted RUL distribution.
- Stochastic simulations from SoC(t) produce the reference RUL distribution.
- CRPS measures the distance between the two distributions.

Requires a trained HeteroscedasticCNN1D checkpoint from ``train.py``.
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

from qmodem import HeteroscedasticCNN1D  # noqa: E402
from qmodem.metrics import crps  # noqa: E402


def main() -> None:
    np.random.seed(TEST_SEED)

    # Configuration
    N_SIMU = 200  # Stochastic simulations per eval point (reference distribution)
    N_PRED_SAMPLES = 200  # Samples from predicted Gaussian
    N_EVAL_POINTS = 15  # Number of evaluation time points along the trajectory

    # Directories
    root_dir, _, METADATA_DIR = get_run_dirs("het_cnn_train", create=False)
    ckpt_dir = root_dir / "checkpoints"
    output_dir = Path("saved/het_cnn_crps_over_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(METADATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    window_size = metadata["training_params"]["window_size"]
    y_max_train = metadata["scaling_params"]["y_max"]
    dt = metadata["simulator_config"]["dt"]

    print("=" * 70)
    print("CRPS Over Time — Heteroscedastic CNN")
    print("=" * 70)
    print(f"Stochastic sims per eval point: {N_SIMU}")
    print(f"Predicted Gaussian samples: {N_PRED_SAMPLES}")
    print(f"Evaluation points: {N_EVAL_POINTS}")
    print()

    # Load trained model
    print("Loading trained heteroscedastic CNN model...")
    model = restore_model_from_checkpoint(
        ckpt_dir / "trained_state",
        lambda: HeteroscedasticCNN1D(**metadata["model_params"], rngs=nnx.Rngs(0)),
    )
    model.eval()
    print("Model loaded successfully.")
    print()

    # Run a single stochastic simulation (same noise as training) as the observed
    # trajectory. This provides the voltage windows fed to the CNN and the SoC
    # trajectory used to seed reference simulations.
    print("Running stochastic test simulation...")
    sim_config = metadata["simulator_config"].copy()
    sim_config["N_simu"] = 1
    sim_0 = les.SimulatorSimple(sim_config)
    sim_0.simulate()

    discharge_voltage = sim_0.v_memo.flatten()  # shape (N_t,)
    socs = sim_0.soc_memo.flatten()  # shape (N_t,)
    t_eod = sim_0._t_eods[0]
    N_t = len(discharge_voltage)

    print(f"Trajectory length: {N_t} steps, t_eod={t_eod:.1f}s")
    print()

    # Select evaluation time indices (must have a full window available).
    first_valid = window_size
    eval_indices = np.linspace(first_valid, N_t - 1, N_EVAL_POINTS, dtype=int)

    ts_eval = []
    crps_values = []

    print("Computing CRPS at each evaluation point...")
    for k, idx in enumerate(eval_indices):
        t = idx * dt
        ts_eval.append(t)

        # --- Predicted distribution (CNN) ---
        start = idx - window_size
        window = discharge_voltage[start:idx].reshape(1, -1)
        pred = model(jnp.expand_dims(window, 0))[0]  # shape (2,)
        mu = float(pred[0]) * y_max_train
        var = float(pred[1]) * y_max_train**2
        std = np.sqrt(max(var, 1e-12))
        pred_samples = np.clip(np.random.normal(mu, std, size=N_PRED_SAMPLES), 0, None)

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
            f"mu={mu:7.1f} | std={std:6.1f} | CRPS={crps_val:.3f}"
        )

    print()

    # Plot CRPS over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts_eval, crps_values, marker="o", linewidth=2)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CRPS [s]")
    ax.set_title("CRPS Over Time — Heteroscedastic CNN vs Simulator")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "crps_over_time.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_dir / 'crps_over_time.png'}")
    plt.show()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
