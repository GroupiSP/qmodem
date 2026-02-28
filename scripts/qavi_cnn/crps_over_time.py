"""Plot CRPS over time for QAVI CNN RUL predictions.

At each evaluation time t along a stochastic discharge trajectory:
- The CNN receives the voltage window ending at t.
- N_WEIGHT_SAMPLES sets of conv weights are generated from PQC circuits.
- Each weight sample produces (mu, var); one sample is drawn from N(mu, sqrt(var))
  to capture both epistemic (weight) and aleatoric (output Gaussian) uncertainty.
- Stochastic simulations from SoC(t) produce the reference RUL distribution.
- CRPS measures the distance between the two distributions.

Requires a trained QAVI CNN checkpoint from ``train.py``.

Run::

    uv run python scripts/qavi_cnn/crps_over_time.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _seeds import TEST_SEED  # noqa: E402
from _shared import get_run_dirs  # noqa: E402

from qmodem import QAVICNN1D  # noqa: E402
from qmodem.metrics import crps  # noqa: E402


# ---------------------------------------------------------------------------
# PQC circuit builder (must match training architecture)
# ---------------------------------------------------------------------------
def _make_pqc(n_qubits: int, n_layers: int):
    """Build a variational quantum circuit for one conv filter."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax")
    def circuit(params: jax.Array, z: float) -> list:
        for i in range(n_qubits):
            qml.RY(z, wires=i)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qml.RY(params[layer, q, 0], wires=q)
                qml.RZ(params[layer, q, 1], wires=q)
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


# ---------------------------------------------------------------------------
# PostProcessor (must match training)
# ---------------------------------------------------------------------------
class PostProcessor(nnx.Module):
    """Linear map from qubit expectation values to conv filter weights."""

    def __init__(self, n_qubits: int, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(n_qubits, n_qubits, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Restore helpers
# ---------------------------------------------------------------------------
def load_all_components(
    ckpt_dir: Path, metadata: dict
) -> tuple[tuple, tuple, QAVICNN1D]:
    """Load PQC params, PostProcessors, and QAVICNN1D model from checkpoint."""
    import orbax.checkpoint as ocp

    n_filters = metadata["model_params"]["n_filters"]
    n_qubits = metadata["pqc_params"]["n_qubits"]

    # PQC parameters
    pqc_data = np.load(ckpt_dir / "pqc_params.npz")
    q_params_list = tuple(jnp.array(pqc_data[f"filter_{i}"]) for i in range(n_filters))

    # PostProcessor states
    with open(ckpt_dir / "pp_states.pkl", "rb") as f:
        pp_states_saved = pickle.load(f)

    pp_list = []
    for i, saved_state in enumerate(pp_states_saved):
        pp = PostProcessor(n_qubits, rngs=nnx.Rngs(params=0))
        nnx.update(pp, saved_state)
        pp_list.append(pp)
    pp_list = tuple(pp_list)

    # QAVICNN1D model
    model = QAVICNN1D(rngs=nnx.Rngs(params=0), **metadata["model_params"])
    checkpointer = ocp.StandardCheckpointer()
    target_state = nnx.state(model)
    state_restored = checkpointer.restore(ckpt_dir / "model_state", target=target_state)
    nnx.update(model, state_restored)

    return q_params_list, pp_list, model


def main() -> None:
    np.random.seed(TEST_SEED)

    # Configuration
    N_SIMU = 200
    N_WEIGHT_SAMPLES = 100
    N_EVAL_POINTS = 15

    # Directories
    root_dir, _, METADATA_DIR = get_run_dirs("qavi_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"
    output_dir = Path("saved/qavi_cnn/crps_over_time")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(METADATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    n_filters = metadata["model_params"]["n_filters"]
    kernel_size = metadata["model_params"]["kernel_size"]
    n_qubits = metadata["pqc_params"]["n_qubits"]
    n_pqc_layers = metadata["pqc_params"]["n_pqc_layers"]
    window_size = metadata["training_params"]["window_size"]
    y_max_train = metadata["scaling_params"]["y_max"]
    dt = metadata["simulator_config"]["dt"]

    print("=" * 70)
    print("CRPS Over Time — QAVI CNN")
    print("=" * 70)
    print(f"Stochastic sims per eval point: {N_SIMU}")
    print(f"Weight samples: {N_WEIGHT_SAMPLES}")
    print(f"Evaluation points: {N_EVAL_POINTS}")
    print()

    # Build PQC circuits
    pqc_circuits = [_make_pqc(n_qubits, n_pqc_layers) for _ in range(n_filters)]
    batched_circuits = [jax.vmap(c, in_axes=(None, 0)) for c in pqc_circuits]

    # Load trained model
    print("Loading trained QAVI CNN model...")
    q_params_list, pp_list, model = load_all_components(ckpt_dir, metadata)
    print("Model loaded successfully.")
    print()

    # Run a single stochastic simulation as the observed trajectory.
    print("Running stochastic test simulation...")
    sim_config = metadata["simulator_config"].copy()
    sim_config["N_simu"] = 1
    sim_0 = les.SimulatorSimple(sim_config)
    sim_0.simulate()

    discharge_voltage = sim_0.v_memo.flatten()
    socs = sim_0.soc_memo.flatten()
    N_t = len(discharge_voltage)

    print(f"Trajectory length: {N_t} steps")
    print()

    # Select evaluation time indices
    first_valid = window_size
    eval_indices = np.linspace(first_valid, N_t - 1, N_EVAL_POINTS, dtype=int)

    base_key = jax.random.PRNGKey(42)
    ts_eval = []
    crps_values = []

    print("Computing CRPS at each evaluation point...")
    for k, idx in enumerate(eval_indices):
        t = idx * dt
        ts_eval.append(t)

        # --- Predicted distribution (QAVI CNN) ---
        start = idx - window_size
        window = discharge_voltage[start:idx].reshape(1, -1)
        x_input = jnp.expand_dims(window, 0)

        pred_samples = []
        for i in range(N_WEIGHT_SAMPLES):
            key = jax.random.fold_in(base_key, k * N_WEIGHT_SAMPLES + i)
            z = jax.random.uniform(key, (1,), minval=0.0, maxval=2.0 * jnp.pi)

            kernels_i = []
            biases_i = []
            for f_idx in range(n_filters):
                expvals = batched_circuits[f_idx](q_params_list[f_idx], z)
                expvals = jnp.stack(expvals, axis=-1)
                weights = pp_list[f_idx](expvals)
                kernels_i.append(weights[0, :kernel_size])
                biases_i.append(weights[0, kernel_size:])

            kernel = jnp.stack(kernels_i, axis=-1)[:, jnp.newaxis, :]  # (5, 1, 4)
            bias = jnp.concatenate(biases_i)  # (4,)

            pred = model(x_input, kernel, bias)[0]
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
    ax.set_title("CRPS Over Time — QAVI CNN vs Simulator")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / "crps_over_time.png", dpi=150, bbox_inches="tight")
    print(f"Figure saved to {output_dir / 'crps_over_time.png'}")
    plt.show()

    print()
    print("Done!")


if __name__ == "__main__":
    main()
