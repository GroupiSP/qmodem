"""Train a QAVI CNN on time-windowed battery discharge data for RUL prediction.

This script demonstrates:
- Quantum Adversarial Variational Inference (QAVI) for CNN weight generation
- 4 PQC generators (one per conv filter, 6 qubits each) produce conv weights
- Adversarial training with a classical MLP discriminator
- Heteroscedastic output (mean + variance) via GaussianBlock

Architecture:
- **Quantum generators**: 4 PennyLane variational circuits (6 qubits, configurable
  layers) on ``default.qubit``.  First 5 qubit expectation values → kernel weights,
  6th → bias.
- **Post-processors**: 4 Flax NNX ``Linear(6, 6)`` layers, one per PQC.
- **DDM**: :class:`~qmodem.QAVICNN1D` — functional Conv1D (external weights) →
  GELU → Global Average Pooling → GaussianBlock.
- **Discriminator**: Flax NNX MLP with LeakyReLU + sigmoid.
- **Likelihood**: Gaussian negative log-likelihood.

Run::

    uv run python scripts/qavi_cnn/train.py
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pennylane as qml
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _seeds import TRAIN_SEED  # noqa: E402
from _shared import (  # noqa: E402
    create_battery_and_policy,
    get_run_dirs,
    make_simulator_config,
)
from grain import DataLoader  # noqa: E402
from grain.samplers import IndexSampler  # noqa: E402
from grain.transforms import Batch  # noqa: E402

from qmodem import QAVICNN1D, BatterySimulationTimeWindowSource  # noqa: E402

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
# PQC
N_QUBITS = 6
N_PQC_LAYERS = 1

# CNN
N_FILTERS = 4
KERNEL_SIZE = 5

# Training
LR_GEN = 0.01
LR_DISC = 0.001
N_EPOCHS = 500
BATCH_SIZE = 32
BATCH_W = 32  # weight samples per step
PATIENCE = 50
PRINT_EVERY = 10

# Data
N_SIMU_TRAIN = 100
N_SIMU_VAL = 20
WINDOW_SIZE = 30
STRIDE = 15

# Battery simulation
CURRENT_AMPLITUDE = -2.8 * 0.75
V_CUT = 2.5
DT = 10.0
OMEGA_STD = 3e-3
ETA_STD = 3e-2

# Discriminator
DISC_HIDDEN = 64

SEED = TRAIN_SEED
EPS = 1e-7


# ---------------------------------------------------------------------------
# PQC circuits (PennyLane)
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


# One circuit per filter
pqc_circuits = [_make_pqc(N_QUBITS, N_PQC_LAYERS) for _ in range(N_FILTERS)]
batched_circuits = [jax.vmap(c, in_axes=(None, 0)) for c in pqc_circuits]


# ---------------------------------------------------------------------------
# Post-processor (Flax NNX)
# ---------------------------------------------------------------------------
class PostProcessor(nnx.Module):
    """Linear map from qubit expectation values to conv filter weights."""

    def __init__(self, n_qubits: int, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(n_qubits, n_qubits, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Discriminator (Flax NNX)
# ---------------------------------------------------------------------------
class Discriminator(nnx.Module):
    """MLP discriminator: input_dim → hidden → hidden → 1."""

    def __init__(self, input_dim: int, hidden: int = 64, *, rngs: nnx.Rngs) -> None:
        self.l1 = nnx.Linear(input_dim, hidden, rngs=rngs)
        self.l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.l3 = nnx.Linear(hidden, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.leaky_relu(self.l1(x), negative_slope=0.2)
        x = nnx.leaky_relu(self.l2(x), negative_slope=0.2)
        return nnx.sigmoid(self.l3(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# Generator forward: PQCs → post-processors → assemble kernel/bias
# ---------------------------------------------------------------------------
def generator_forward(
    q_params_list: tuple[jax.Array, ...],
    pp_list: tuple[PostProcessor, ...],
    z_batch: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Run 4 PQC generators + post-processors and assemble conv weights.

    Args:
        q_params_list: Tuple of 4 PQC param arrays, each
            ``(n_layers, n_qubits, 2)``.
        pp_list: Tuple of 4 PostProcessor modules.
        z_batch: Latent variables ``(batch_w,)``.

    Returns:
        kernel ``(batch_w, kernel_size, 1, n_filters)`` and
        bias ``(batch_w, n_filters)``.
    """
    kernels = []
    biases = []
    for i in range(N_FILTERS):
        expvals = batched_circuits[i](q_params_list[i], z_batch)  # list of (batch_w,)
        expvals = jnp.stack(expvals, axis=-1)  # (batch_w, n_qubits)
        weights = pp_list[i](expvals)  # (batch_w, n_qubits)
        # First 5 → kernel weights, last 1 → bias
        kernels.append(weights[:, :KERNEL_SIZE])  # (batch_w, 5)
        biases.append(weights[:, KERNEL_SIZE:])  # (batch_w, 1)

    # Assemble: kernel (batch_w, kernel_size, 1, n_filters)
    kernel = jnp.stack(kernels, axis=-1)  # (batch_w, 5, 4)
    kernel = kernel[:, :, jnp.newaxis, :]  # (batch_w, 5, 1, 4)
    bias = jnp.concatenate(biases, axis=-1)  # (batch_w, 4)
    return kernel, bias


# ---------------------------------------------------------------------------
# DDM forward: batched over weight samples
# ---------------------------------------------------------------------------
def ddm_forward_single(
    x_batch: jax.Array,
    kernel: jax.Array,
    bias: jax.Array,
    model: QAVICNN1D,
) -> jax.Array:
    """Run QAVICNN1D with one set of conv weights on a data batch.

    Returns:
        Predictions ``(batch_size, 2)`` with ``[mu, var]``.
    """
    return model(x_batch, kernel, bias)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def disc_loss_fn(
    disc: Discriminator,
    x_batch: jax.Array,
    y_batch: jax.Array,
    mu_preds: jax.Array,
) -> jax.Array:
    """Discriminator loss.

    Args:
        disc: Discriminator module.
        x_batch: Voltage windows ``(batch_size, 1, window_size)``.
        y_batch: True RUL values ``(batch_size,)``.
        mu_preds: Predicted means ``(batch_w, batch_size)``.
    """
    # Flatten x for discriminator: (batch_size, window_size)
    x_flat = x_batch.squeeze(1)

    # Real pairs: (batch_size, window_size + 1)
    real_pairs = jnp.concatenate([x_flat, y_batch[:, jnp.newaxis]], axis=-1)
    d_real = disc(real_pairs)
    loss_real = -jnp.mean(jnp.log(d_real + EPS))

    # Fake pairs: (batch_w, batch_size, window_size + 1)
    x_exp = jnp.broadcast_to(
        x_flat[jnp.newaxis, :, :], (mu_preds.shape[0],) + x_flat.shape
    )
    fake_pairs = jnp.concatenate([x_exp, mu_preds[:, :, jnp.newaxis]], axis=-1).reshape(
        -1, x_flat.shape[-1] + 1
    )
    d_fake = disc(fake_pairs).reshape(mu_preds.shape)
    loss_fake = -jnp.mean(jnp.log(1.0 - d_fake + EPS))

    return loss_real + loss_fake


def gen_loss_fn(
    q_params_list: tuple[jax.Array, ...],
    pp_states: tuple,
    pp_graphdefs: tuple,
    model_state: nnx.State,
    model_graphdef: nnx.GraphDef,
    disc: Discriminator,
    z_batch: jax.Array,
    x_batch: jax.Array,
    y_batch: jax.Array,
) -> jax.Array:
    """Generator loss: adversarial term + Gaussian NLL likelihood.

    Args:
        q_params_list: Tuple of 4 PQC param arrays.
        pp_states: Tuple of 4 PostProcessor states.
        pp_graphdefs: Tuple of 4 PostProcessor graph defs.
        model_state: QAVICNN1D state (GaussianBlock params).
        model_graphdef: QAVICNN1D graph def.
        disc: Discriminator module (frozen for this step).
        z_batch: Latent variables ``(batch_w,)``.
        x_batch: Voltage windows ``(batch_size, 1, window_size)``.
        y_batch: True RUL values ``(batch_size,)``.
    """
    # Reconstruct PostProcessor modules
    pp_list = tuple(nnx.merge(gd, st) for gd, st in zip(pp_graphdefs, pp_states))
    model = nnx.merge(model_graphdef, model_state)

    kernels, biases = generator_forward(q_params_list, pp_list, z_batch)

    # Run DDM for each weight sample
    def _single(kernel, bias):
        return model(x_batch, kernel, bias)

    outputs = jax.vmap(_single)(kernels, biases)  # (batch_w, batch_size, 2)
    mu_preds = outputs[:, :, 0]
    var_preds = jnp.clip(outputs[:, :, 1], min=1e-6)

    # --- Adversarial term ---
    x_flat = x_batch.squeeze(1)
    x_exp = jnp.broadcast_to(
        x_flat[jnp.newaxis, :, :], (mu_preds.shape[0],) + x_flat.shape
    )
    fake_pairs = jnp.concatenate([x_exp, mu_preds[:, :, jnp.newaxis]], axis=-1).reshape(
        -1, x_flat.shape[-1] + 1
    )
    d_fake = disc(fake_pairs).reshape(mu_preds.shape)
    d_clamped = jnp.clip(d_fake, EPS, 1.0 - EPS)
    logits = jnp.log(d_clamped / (1.0 - d_clamped))
    adv_loss = -jnp.mean(logits)

    # --- Likelihood term (Gaussian NLL) ---
    y_exp = jnp.broadcast_to(y_batch[jnp.newaxis, :], mu_preds.shape)
    nll = jnp.mean(
        0.5 * jnp.log(var_preds) + 0.5 * jnp.square(y_exp - mu_preds) / var_preds
    )

    return adv_loss + nll


# ---------------------------------------------------------------------------
# Training step factories
# ---------------------------------------------------------------------------
def make_disc_step(disc_optimizer: optax.GradientTransformation):
    """Return a JIT-compiled discriminator update step."""

    @jax.jit
    def disc_step(
        disc: Discriminator,
        disc_opt_state: optax.OptState,
        x_batch: jax.Array,
        y_batch: jax.Array,
        mu_preds: jax.Array,
    ) -> tuple[jax.Array, Discriminator, optax.OptState]:
        graphdef, state = nnx.split(disc)

        def loss_wrapper(state):
            d = nnx.merge(graphdef, state)
            return disc_loss_fn(d, x_batch, y_batch, mu_preds)

        loss, grads = jax.value_and_grad(loss_wrapper)(state)
        updates, new_opt_state = disc_optimizer.update(grads, disc_opt_state, state)
        new_state = optax.apply_updates(state, updates)
        new_disc = nnx.merge(graphdef, new_state)
        return loss, new_disc, new_opt_state

    return disc_step


def make_gen_step(gen_optimizer: optax.GradientTransformation):
    """Return a JIT-compiled generator update step."""

    @jax.jit
    def gen_step(
        q_params_list: tuple[jax.Array, ...],
        pp_list: tuple[PostProcessor, ...],
        model: QAVICNN1D,
        gen_opt_state: optax.OptState,
        disc: Discriminator,
        z_batch: jax.Array,
        x_batch: jax.Array,
        y_batch: jax.Array,
    ) -> tuple[
        jax.Array,
        tuple[jax.Array, ...],
        tuple[PostProcessor, ...],
        QAVICNN1D,
        optax.OptState,
    ]:
        # Split NNX modules into graphdefs + states for differentiability
        pp_splits = [nnx.split(pp) for pp in pp_list]
        pp_graphdefs = tuple(s[0] for s in pp_splits)
        pp_states = tuple(s[1] for s in pp_splits)

        model_graphdef, model_state = nnx.split(model)

        def loss_wrapper(q_params_list, pp_states, model_state):
            return gen_loss_fn(
                q_params_list,
                pp_states,
                pp_graphdefs,
                model_state,
                model_graphdef,
                disc,
                z_batch,
                x_batch,
                y_batch,
            )

        loss, grads = jax.value_and_grad(loss_wrapper, argnums=(0, 1, 2))(
            q_params_list, pp_states, model_state
        )
        q_grads, pp_grads, m_grads = grads

        gen_params = (q_params_list, pp_states, model_state)
        gen_grads_all = (q_grads, pp_grads, m_grads)
        updates, new_opt_state = gen_optimizer.update(
            gen_grads_all, gen_opt_state, gen_params
        )
        new_q, new_pp_st, new_m_st = optax.apply_updates(gen_params, updates)

        new_pp_list = tuple(
            nnx.merge(gd, st) for gd, st in zip(pp_graphdefs, new_pp_st)
        )
        new_model = nnx.merge(model_graphdef, new_m_st)

        return loss, new_q, new_pp_list, new_model, new_opt_state

    return gen_step


# ---------------------------------------------------------------------------
# JIT'd helpers for disc pre-computation and validation
# ---------------------------------------------------------------------------
@jax.jit
def compute_mu_preds(
    q_params_list: tuple[jax.Array, ...],
    pp_list: tuple[PostProcessor, ...],
    model: QAVICNN1D,
    z_batch: jax.Array,
    x_batch: jax.Array,
) -> jax.Array:
    """JIT'd computation of mu predictions for the discriminator step."""
    kernels, biases = generator_forward(q_params_list, pp_list, z_batch)

    def _single(kernel, bias):
        return model(x_batch, kernel, bias)

    outputs = jax.vmap(_single)(kernels, biases)  # (batch_w, batch_size, 2)
    return outputs[:, :, 0]


@jax.jit
def val_step(
    q_params_list: tuple[jax.Array, ...],
    pp_list: tuple[PostProcessor, ...],
    model: QAVICNN1D,
    disc: Discriminator,
    z_batch: jax.Array,
    x_batch: jax.Array,
    y_batch: jax.Array,
) -> jax.Array:
    """JIT'd validation step computing generator loss."""
    pp_splits = [nnx.split(pp) for pp in pp_list]
    pp_graphdefs = tuple(s[0] for s in pp_splits)
    pp_states = tuple(s[1] for s in pp_splits)
    model_graphdef, model_state = nnx.split(model)

    return gen_loss_fn(
        q_params_list,
        pp_states,
        pp_graphdefs,
        model_state,
        model_graphdef,
        disc,
        z_batch,
        x_batch,
        y_batch,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Train a QAVI CNN on time-windowed battery data."""
    np.random.seed(SEED)

    _root_dir, CHECKPOINT_DIR, METADATA_DIR = get_run_dirs(
        "qavi_cnn/train", create=True
    )

    battery, discharge_policy = create_battery_and_policy(CURRENT_AMPLITUDE)

    print("=" * 70)
    print("QAVI CNN Training on Time-Windowed Battery Data")
    print("=" * 70)
    print(f"PQC: {N_QUBITS} qubits, {N_PQC_LAYERS} layers, {N_FILTERS} filters")
    print(f"Window size: {WINDOW_SIZE}, Stride: {STRIDE}")
    print(f"Train sims: {N_SIMU_TRAIN}, Val sims: {N_SIMU_VAL}")
    print(f"Weight samples per step: {BATCH_W}")
    print()

    # --- Data ---
    shared_sim_config = make_simulator_config(
        n_simu=1,
        v_cut=V_CUT,
        soc_0=1.0,
        dt=DT,
        omega_std=OMEGA_STD,
        eta_std=ETA_STD,
        discharge_policy=discharge_policy,
        battery=battery,
    )

    print("Creating training dataset...")
    ds_train = BatterySimulationTimeWindowSource(
        shared_sim_config, n_hists=N_SIMU_TRAIN, window_size=WINDOW_SIZE, stride=STRIDE
    )
    print(f"Total training windows: {len(ds_train)}")

    print("Creating validation dataset...")
    ds_val = BatterySimulationTimeWindowSource(
        shared_sim_config, n_hists=N_SIMU_VAL, window_size=WINDOW_SIZE, stride=STRIDE
    )
    print(f"Total validation windows: {len(ds_val)}")
    print()

    sampler_train = IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=True, seed=42
    )
    dataloader_train = DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[Batch(batch_size=BATCH_SIZE, drop_remainder=True)],
        worker_count=0,
    )

    sampler_val = IndexSampler(
        num_records=len(ds_val), num_epochs=1, shuffle=False, seed=0
    )
    dataloader_val = DataLoader(
        data_source=ds_val,
        sampler=sampler_val,
        operations=[Batch(batch_size=BATCH_SIZE, drop_remainder=True)],
        worker_count=0,
    )

    # --- Initialise components ---
    rng = jax.random.PRNGKey(SEED)

    # PQC parameters: 4 sets of (n_layers, n_qubits, 2)
    q_params_list = []
    for i in range(N_FILTERS):
        rng, k = jax.random.split(rng)
        q_params_list.append(jax.random.normal(k, (N_PQC_LAYERS, N_QUBITS, 2)) * 0.1)
    q_params_list = tuple(q_params_list)

    # PostProcessors: 4 Linear(6, 6)
    pp_list = tuple(
        PostProcessor(N_QUBITS, rngs=nnx.Rngs(params=SEED + i))
        for i in range(N_FILTERS)
    )

    # DDM model (only GaussianBlock is trainable)
    model = QAVICNN1D(
        n_filters=N_FILTERS,
        kernel_size=KERNEL_SIZE,
        rngs=nnx.Rngs(params=SEED + N_FILTERS),
    )

    # Discriminator: input_dim = window_size + 1 (voltage window + RUL value)
    disc_input_dim = WINDOW_SIZE + 1
    disc = Discriminator(
        disc_input_dim, hidden=DISC_HIDDEN, rngs=nnx.Rngs(params=SEED + N_FILTERS + 1)
    )

    # Count parameters
    n_pqc_params = sum(p.size for p in q_params_list)
    n_pp_params = sum(
        p.size for pp in pp_list for p in jax.tree.leaves(nnx.state(pp, nnx.Param))
    )
    n_model_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    n_disc_params = sum(p.size for p in jax.tree.leaves(nnx.state(disc, nnx.Param)))
    print(f"PQC parameters: {n_pqc_params}")
    print(f"PostProcessor parameters: {n_pp_params}")
    print(f"Model (GaussianBlock) parameters: {n_model_params}")
    print(f"Discriminator parameters: {n_disc_params}")
    print(f"Total generator parameters: {n_pqc_params + n_pp_params + n_model_params}")
    print()

    # --- Optimizers ---
    gen_optimizer = optax.adam(LR_GEN)
    disc_optimizer = optax.adam(LR_DISC)

    # Init optimizer states
    pp_states_init = tuple(nnx.split(pp)[1] for pp in pp_list)
    _, model_state_init = nnx.split(model)
    gen_opt_state = gen_optimizer.init(
        (q_params_list, pp_states_init, model_state_init)
    )

    _, disc_state_init = nnx.split(disc)
    disc_opt_state = disc_optimizer.init(disc_state_init)

    disc_step = make_disc_step(disc_optimizer)
    gen_step = make_gen_step(gen_optimizer)

    # --- Training loop ---
    print("Starting QAVI training...")
    print("=" * 70)

    base_key = jax.random.PRNGKey(0)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    t0 = time.time()

    for epoch in range(N_EPOCHS):
        train_g_losses = []
        train_d_losses = []

        for batch in dataloader_train:
            x_batch, y_batch = batch

            # Sample latent z
            key = jax.random.fold_in(base_key, global_step)
            k1, k2 = jax.random.split(key)
            z_batch = jax.random.uniform(
                k1, (BATCH_W,), minval=0.0, maxval=2.0 * jnp.pi
            )

            # Generate mu predictions for discriminator step (JIT'd)
            mu_preds = compute_mu_preds(q_params_list, pp_list, model, z_batch, x_batch)

            # Discriminator step
            d_loss, disc, disc_opt_state = disc_step(
                disc, disc_opt_state, x_batch, y_batch, mu_preds
            )

            # Generator step (fresh z)
            z_batch2 = jax.random.uniform(
                k2, (BATCH_W,), minval=0.0, maxval=2.0 * jnp.pi
            )
            g_loss, q_params_list, pp_list, model, gen_opt_state = gen_step(
                q_params_list,
                pp_list,
                model,
                gen_opt_state,
                disc,
                z_batch2,
                x_batch,
                y_batch,
            )

            train_g_losses.append(float(g_loss))
            train_d_losses.append(float(d_loss))
            global_step += 1

        mean_g = np.mean(train_g_losses)
        mean_d = np.mean(train_d_losses)

        # --- Validation (generator loss only) ---
        val_losses = []
        for batch in dataloader_val:
            x_batch, y_batch = batch
            key = jax.random.fold_in(base_key, global_step)
            z_batch = jax.random.uniform(
                key, (BATCH_W,), minval=0.0, maxval=2.0 * jnp.pi
            )

            v_loss = val_step(
                q_params_list, pp_list, model, disc, z_batch, x_batch, y_batch
            )
            val_losses.append(float(v_loss))
            global_step += 1

        mean_val = np.mean(val_losses)

        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1:3d}/{N_EPOCHS} | "
                f"G_loss: {mean_g:.4f} | D_loss: {mean_d:.4f} | "
                f"Val: {mean_val:.4f} | [{elapsed:.1f}s]"
            )

        # Early stopping on validation loss
        if mean_val < best_val_loss - 1e-4:
            best_val_loss = mean_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"Training complete in {elapsed:.1f}s! Best val loss: {best_val_loss:.6f}")
    print()

    # --- Save artifacts ---
    print("Saving checkpoint...")
    import orbax.checkpoint as ocp

    # Erase and recreate checkpoint directory for a clean save
    ckpt_base = ocp.test_utils.erase_and_create_empty(CHECKPOINT_DIR)

    # PQC parameters
    np.savez(
        ckpt_base / "pqc_params.npz",
        **{f"filter_{i}": np.array(q_params_list[i]) for i in range(N_FILTERS)},
    )

    # PostProcessor states
    pp_states_save = [nnx.state(pp, nnx.Param) for pp in pp_list]
    with open(ckpt_base / "pp_states.pkl", "wb") as f:
        pickle.dump(pp_states_save, f)

    # Model state (GaussianBlock)
    checkpointer = ocp.StandardCheckpointer()
    _, model_state_final = nnx.split(model)
    checkpointer.save(ckpt_base / "model_state", model_state_final)
    time.sleep(0.5)

    print(f"Checkpoint saved to {CHECKPOINT_DIR}")

    # Metadata
    metadata = {
        "simulator_config": shared_sim_config,
        "training_params": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "n_simu_train": N_SIMU_TRAIN,
            "n_simu_val": N_SIMU_VAL,
            "batch_w": BATCH_W,
        },
        "model_params": {
            "n_filters": N_FILTERS,
            "kernel_size": KERNEL_SIZE,
        },
        "pqc_params": {
            "n_qubits": N_QUBITS,
            "n_pqc_layers": N_PQC_LAYERS,
        },
        "scaling_params": {
            "y_max": ds_train.y_max.item(),
        },
    }
    with open(METADATA_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"Metadata saved to {METADATA_DIR}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
