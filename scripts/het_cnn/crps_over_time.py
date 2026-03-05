"""Plot CRPS over time for heteroscedastic CNN RUL predictions.

At each evaluation time t along a stochastic discharge trajectory:
- The CNN receives the voltage window ending at t and outputs (mu, var).
- Samples from N(mu, var), clipped at zero, form the predicted RUL distribution.
- Pre-generated stochastic simulations from SoC(t) provide the reference RUL
  distribution.
- CRPS measures the distance between the two distributions.

Requires a trained HeteroscedasticCNN1D checkpoint from ``train.py``
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
    restore_model_from_checkpoint,
)

from qmodem import HeteroscedasticCNN1D  # noqa: E402
from qmodem.metrics import crps  # noqa: E402


def main() -> None:
    np.random.seed(TEST_SEED)

    # Configuration
    N_PRED_SAMPLES = 500  # Samples from predicted Gaussian

    # Directories
    root_dir, _, METADATA_DIR = get_run_dirs("het_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"
    output_dir = Path("saved/het_cnn/crps_over_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(METADATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    window_size = metadata["training_params"]["window_size"]
    y_max_train = metadata["scaling_params"]["y_max"]

    print("=" * 70)
    print("CRPS Over Time — Heteroscedastic CNN")
    print("=" * 70)
    print(f"Predicted Gaussian samples: {N_PRED_SAMPLES}")
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

    # Load pre-generated test case
    print("Loading test case data...")
    test_data = np.load("data/test_case_0.npz")
    discharge_voltage = test_data["voltage"]
    dt = float(test_data["dt"])
    eval_indices = test_data["eval_indices"]
    ref_t_eods = test_data["ref_t_eods"]
    N_t = len(discharge_voltage)
    N_EVAL_POINTS = len(eval_indices)

    print(f"Trajectory length: {N_t} steps")
    print(f"Evaluation points: {N_EVAL_POINTS}")
    print()

    ts_eval = []
    crps_values = []

    print("Computing CRPS at each evaluation point...")
    for k, idx in enumerate(eval_indices):
        t = idx * dt

        # --- Predicted distribution (CNN) ---
        start = idx - window_size
        if start < 0:
            continue
        ts_eval.append(t)
        window = discharge_voltage[start:idx].reshape(1, -1)
        pred = model(jnp.expand_dims(window, 0))[0]  # shape (2,)
        mu = float(pred[0]) * y_max_train
        var = float(pred[1]) * y_max_train**2
        std = np.sqrt(max(var, 1e-12))
        pred_samples = np.clip(np.random.normal(mu, std, size=N_PRED_SAMPLES), 0, None)

        # --- Reference distribution (pre-generated) ---
        ref_samples = ref_t_eods[k]

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
