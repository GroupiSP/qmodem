"""Compare QAVI CNN predictions with simulator predictions for battery RUL.

This script:
- Runs a deterministic simulation to get voltage trajectory
- Runs stochastic simulations from SOCs after the first time window
- Uses trained QAVI CNN to predict RUL with uncertainty
- Point prediction: averages μ_out across M weight samples (Bayesian predictive mean)
- Uncertainty: draws one sample from N(μ_out, σ²_out) per weight sample to capture
  both epistemic and aleatoric uncertainty
- Compares predictions using 95% CI plots

Requires a trained QAVI CNN checkpoint from ``train.py``.

Run::

    uv run python scripts/qavi_cnn/rul.py
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

    # QAVICNN1D model (GaussianBlock state)
    model = QAVICNN1D(rngs=nnx.Rngs(params=0), **metadata["model_params"])
    checkpointer = ocp.StandardCheckpointer()
    target_state = nnx.state(model)
    state_restored = checkpointer.restore(ckpt_dir / "model_state", target=target_state)
    nnx.update(model, state_restored)

    return q_params_list, pp_list, model


def main() -> None:
    np.random.seed(TEST_SEED)

    # Directories
    root_dir, _, METADATA_DIR = get_run_dirs("qavi_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"
    output_dir = Path("saved/qavi_cnn/rul")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(METADATA_DIR / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    N_WEIGHT_SAMPLES = 500
    N_SIMU = 500
    N_INTERMEDIATE_SOCs = 200

    n_filters = metadata["model_params"]["n_filters"]
    kernel_size = metadata["model_params"]["kernel_size"]
    n_qubits = metadata["pqc_params"]["n_qubits"]
    n_pqc_layers = metadata["pqc_params"]["n_pqc_layers"]

    print("=" * 70)
    print("QAVI CNN RUL Prediction with Uncertainty")
    print("=" * 70)
    print(f"Number of weight samples: {N_WEIGHT_SAMPLES}")
    print(f"Number of stochastic simulations: {N_SIMU}")
    print()

    # Build PQC circuits
    pqc_circuits = [_make_pqc(n_qubits, n_pqc_layers) for _ in range(n_filters)]
    batched_circuits = [jax.vmap(c, in_axes=(None, 0)) for c in pqc_circuits]

    def generator_forward(q_params_list, pp_list, z_batch):
        """PQCs → post-processors → assemble kernel/bias (batched)."""
        kernels = []
        biases = []
        for i in range(n_filters):
            expvals = batched_circuits[i](q_params_list[i], z_batch)
            expvals = jnp.stack(expvals, axis=-1)
            weights = pp_list[i](expvals)
            kernels.append(weights[:, :kernel_size])
            biases.append(weights[:, kernel_size:])
        kernel = jnp.stack(kernels, axis=-1)[:, :, jnp.newaxis, :]
        bias = jnp.concatenate(biases, axis=-1)
        return kernel, bias

    @jax.jit
    def predict_window(q_params_list, pp_list, model, z_batch, x_input):
        """JIT'd batched prediction: returns (N_WEIGHT_SAMPLES, 2)."""
        kernels, biases = generator_forward(q_params_list, pp_list, z_batch)

        def _single(kernel, bias):
            return model(x_input, kernel, bias)

        outputs = jax.vmap(_single)(kernels, biases)  # (N_W, 1, 2)
        return outputs[:, 0, :]  # (N_W, 2)

    # Load trained components
    print("Loading trained QAVI CNN model...")
    q_params_list, pp_list, model = load_all_components(ckpt_dir, metadata)
    print("Model loaded successfully")
    print()

    # Run a single deterministic simulation
    print("Part 1. Single simulation to get voltage and SoC history...")
    sim_0 = les.SimulatorSimple(metadata["simulator_config"])
    sim_0.simulate()

    dt = metadata["simulator_config"]["dt"]
    N_t = sim_0.v_memo.shape[0]
    discharge_voltage = sim_0.v_memo.flatten()
    window_size = metadata["training_params"]["window_size"]
    stride = metadata["training_params"]["stride"]
    y_max_train = metadata["scaling_params"]["y_max"]

    base_key = jax.random.PRNGKey(42)

    ts_pred = []
    pred_means = []
    pred_lowers = []
    pred_uppers = []

    # Pre-generate all z values for reproducibility
    z_all = jnp.stack(
        [
            jax.random.uniform(
                jax.random.fold_in(base_key, i),
                (),
                minval=0.0,
                maxval=2.0 * jnp.pi,
            )
            for i in range(N_WEIGHT_SAMPLES)
        ]
    )  # (N_WEIGHT_SAMPLES,)

    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        ts_pred.append(end * dt)
        X = discharge_voltage[start:end].reshape(1, -1)
        x_input = jnp.expand_dims(X, 0)

        # Batched prediction: all weight samples in one JIT'd call
        preds = predict_window(
            q_params_list, pp_list, model, z_all, x_input
        )  # (N_WEIGHT_SAMPLES, 2)
        mus = np.array(preds[:, 0]) * y_max_train
        vars_ = np.array(preds[:, 1]) * y_max_train**2
        stds = np.sqrt(np.clip(vars_, 1e-12, None))
        full_samples = np.clip(np.random.normal(mus, stds), 0, None)

        pred_means.append(float(np.mean(mus)))
        pred_lowers.append(float(np.percentile(full_samples, 2.5)))
        pred_uppers.append(float(np.percentile(full_samples, 97.5)))

    # Cover the last part of the trajectory if it doesn't fit a full window
    if end < N_t:
        ts_pred.append(N_t * dt)
        X = discharge_voltage[-window_size:].reshape(1, -1)
        x_input = jnp.expand_dims(X, 0)

        preds = predict_window(q_params_list, pp_list, model, z_all, x_input)
        mus = np.array(preds[:, 0]) * y_max_train
        vars_ = np.array(preds[:, 1]) * y_max_train**2
        stds = np.sqrt(np.clip(vars_, 1e-12, None))
        full_samples = np.clip(np.random.normal(mus, stds), 0, None)

        pred_means.append(float(np.mean(mus)))
        pred_lowers.append(float(np.percentile(full_samples, 2.5)))
        pred_uppers.append(float(np.percentile(full_samples, 97.5)))

    print(
        "Part 2. Running stochastic simulations from intermediate SOCs and comparing "
        "with QAVI CNN predictions..."
    )
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
        ruls_true_lowers.append(np.percentile(t_eods, 2.5))
        ruls_true_uppers.append(np.percentile(t_eods, 97.5))

    # true RUL is linear
    ts_rul_true = np.linspace(0.0, N_t, N_INTERMEDIATE_SOCs) * dt
    ruls_true = sim_0.t_eods[0] - ts_rul_true

    print("Part 3. Plotting results...")
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    ax0.plot(ts_rul_true, ruls_true, label="True RUL")
    ax0.plot(ts_pred, pred_means, label="Predicted RUL (QAVI CNN)", marker="o")
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("RUL [s]")
    ax0.set_title("QAVI CNN RUL Mean Predictions")
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
        label="Predicted RUL (QAVI CNN)",
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
    ax1.set_title("QAVI CNN RUL Predictions with Uncertainty")
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
