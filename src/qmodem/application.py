"""Application-level functions for the QMoDeM workflow.

This module provides the high-level entry points for data generation, model
training, model testing (RUL prediction + CRPS evaluation), and multi-method
comparison.  It is the primary orchestration layer called by
:mod:`qmodem.cli`.

Functions:
    generate_data: Generate train, validation, and test datasets.
    train: Train a model using one of the supported methods.
    test: Evaluate a trained model (RUL prediction + CRPS).
    compare: Compare multiple methods on the same test case.
"""

from __future__ import annotations

import dataclasses
import pickle
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from matplotlib.figure import Figure

from qmodem import (
    MCDCNN1D,
    QAVICNN1D,
    BatterySimulationTimeWindowSource,
    BayesCNN1D,
    FlipoutConv1D,
    HeteroscedasticCNN1D,
    elbo_nll_loss,
    nll_loss,
    nll_loss_mcd,
)
from qmodem.generate import generate_test_data, generate_train_data
from qmodem.metadata import (
    BaseModelParams,
    MCDModelParams,
    PQCParams,
    QAVITrainingMetadata,
    QAVITrainingParams,
    ScalingParams,
    SimulatorConfig,
    TrainingMetadata,
    TrainingParams,
    load_metadata,
    save_metadata,
)
from qmodem.metrics import crps
from qmodem.train import EarlyStopper, train_loop
from qmodem.utils import (
    SHARED_PARAMS,
    TEST_SEED,
    TRAIN_SEED,
    create_battery_and_policy,
    get_run_dirs,
    make_simulator_config,
    restore_model_from_checkpoint,
    restore_model_state,
)

METHODS = ("bayes_cnn", "het_cnn", "mcd_cnn", "qavi_cnn")

METHOD_LABELS: dict[str, str] = {
    "bayes_cnn": "Bayesian CNN (Flipout)",
    "het_cnn": "Heteroscedastic CNN",
    "mcd_cnn": "MC Dropout CNN",
    "qavi_cnn": "QAVI CNN",
}


@dataclasses.dataclass
class TestResult:
    """Prediction outputs produced by a single method on one test case."""

    method_label: str
    ts_rul_true: list[float]
    ruls_true: list[float]
    ruls_true_lowers: list[float]
    ruls_true_uppers: list[float]
    ts_pred: list[float]
    pred_means: list[float]
    pred_lowers: list[float]
    pred_uppers: list[float]
    ts_eval: list[float]
    crps_values: list[float]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _validate_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {', '.join(METHODS)}"
        )


def _load_datasets(
    train_path: str,
    val_path: str,
    window_size: int,
    stride: int,
    normalize: bool,
) -> tuple[BatterySimulationTimeWindowSource, BatterySimulationTimeWindowSource]:
    """Load training and validation time-window datasets from files."""
    print("Loading training dataset...")
    ds_train = BatterySimulationTimeWindowSource.from_file(
        path=train_path, window_size=window_size, stride=stride, normalize=normalize
    )
    print(f"Total training windows: {len(ds_train)}")
    print()

    print("Loading validation dataset...")
    ds_val = BatterySimulationTimeWindowSource.from_file(
        path=val_path, window_size=window_size, stride=stride, normalize=normalize
    )
    print(f"Total validation windows: {len(ds_val)}")
    print()

    return ds_train, ds_val


def _create_dataloaders(
    ds_train: BatterySimulationTimeWindowSource,
    ds_val: BatterySimulationTimeWindowSource,
    batch_size: int,
    *,
    drop_remainder: bool = False,
) -> tuple[Any, Any]:
    """Create Grain DataLoaders for training and validation."""
    from grain import DataLoader
    from grain.samplers import IndexSampler
    from grain.transforms import Batch

    sampler_train = IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=True, seed=42
    )
    dataloader_train = DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
        worker_count=0,
    )

    sampler_val = IndexSampler(
        num_records=len(ds_val), num_epochs=1, shuffle=False, seed=0
    )
    dataloader_val = DataLoader(
        data_source=ds_val,
        sampler=sampler_val,
        operations=[Batch(batch_size=batch_size, drop_remainder=drop_remainder)],
        worker_count=0,
    )

    return dataloader_train, dataloader_val


def _build_sim_config(sim_params: dict[str, Any]) -> SimulatorConfig:
    """Build a JSON-serialisable simulator config from shared simulation parameters."""
    return SimulatorConfig(
        n_simu=1,
        v_cut=sim_params["v_cut"],
        soc_0=1.0,
        dt=sim_params["dt"],
        omega_std=sim_params["omega_std"],
        eta_std=sim_params["eta_std"],
    )


def _build_runtime_sim_config(sim_params: dict[str, Any]) -> dict[str, Any]:
    """Build a full runtime simulator config including battery model and discharge
    policy.

    This config is suitable for passing to ``les.SimulatorSimple`` (data generation).
    For saving metadata to disk, use :func:`_build_sim_config` instead.
    """
    battery, discharge_policy = create_battery_and_policy(
        sim_params["current_amplitude"]
    )
    return make_simulator_config(
        n_simu=1,
        v_cut=sim_params["v_cut"],
        soc_0=1.0,
        dt=sim_params["dt"],
        omega_std=sim_params["omega_std"],
        eta_std=sim_params["eta_std"],
        discharge_policy=discharge_policy,
        battery=battery,
    )


def _save_checkpoint_and_metadata(
    model: nnx.Module,
    checkpoint_dir: Path,
    metadata_dir: Path,
    metadata: TrainingMetadata,
    *,
    params_only: bool = False,
) -> None:
    """Save model checkpoint and metadata as JSON."""
    print("Saving checkpoint...")
    ckpt_dir = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
    checkpointer = ocp.StandardCheckpointer()
    if params_only:
        _graphdef, param_state, _other_state = nnx.split(model, nnx.Param, ...)
        checkpointer.save(ckpt_dir / "trained_state", param_state)
    else:
        _, model_state = nnx.split(model)
        checkpointer.save(ckpt_dir / "trained_state", model_state)
    time.sleep(0.5)
    print(f"Checkpoint saved to {checkpoint_dir}")

    save_metadata(metadata_dir, metadata)
    print(f"Metadata saved to {metadata_dir}")
    print()


# ---------------------------------------------------------------------------
# QAVI-specific components
# ---------------------------------------------------------------------------


class PostProcessor(nnx.Module):
    """Linear map from qubit expectation values to conv filter weights."""

    def __init__(self, n_qubits: int, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(n_qubits, n_qubits, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


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


def _make_pqc(n_qubits: int, n_layers: int):
    """Build a variational quantum circuit for one conv filter."""
    import pennylane as qml

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


def _qavi_generator_forward(
    q_params_list: tuple[jax.Array, ...],
    pp_list: tuple[PostProcessor, ...],
    batched_circuits: list,
    z_batch: jax.Array,
    n_filters: int,
    kernel_size: int,
) -> tuple[jax.Array, jax.Array]:
    """Run PQC generators + post-processors and assemble conv weights."""
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


def _qavi_load_all_components(
    ckpt_dir: Path,
    metadata: QAVITrainingMetadata,
) -> tuple[tuple[jax.Array, ...], tuple[PostProcessor, ...], QAVICNN1D]:
    """Load PQC params, PostProcessors, and QAVICNN1D from checkpoint."""
    n_filters = metadata["model_params"]["n_filters"]
    n_qubits = metadata["pqc_params"]["n_qubits"]

    pqc_data = np.load(ckpt_dir / "pqc_params.npz")
    q_params_list = tuple(jnp.array(pqc_data[f"filter_{i}"]) for i in range(n_filters))

    with open(ckpt_dir / "pp_states.pkl", "rb") as f:
        pp_states_saved = pickle.load(f)

    pp_list = []
    for saved_state in pp_states_saved:
        pp = PostProcessor(n_qubits, rngs=nnx.Rngs(params=0))
        nnx.update(pp, saved_state)
        pp_list.append(pp)
    pp_list = tuple(pp_list)

    model = QAVICNN1D(rngs=nnx.Rngs(params=0), **metadata["model_params"])
    checkpointer = ocp.StandardCheckpointer()
    target_state = nnx.state(model)
    state_restored = checkpointer.restore(ckpt_dir / "model_state", target=target_state)
    nnx.update(model, state_restored)

    return q_params_list, pp_list, model


# ---------------------------------------------------------------------------
# Test plotting helpers
# ---------------------------------------------------------------------------


def populate_rul_ax(
    ax: Any,
    ts_rul_true: np.ndarray | list[float],
    ruls_true: np.ndarray | list[float],
    ruls_true_lowers: list[float],
    ruls_true_uppers: list[float],
    ts_pred: list[float],
    pred_means: list[float],
    pred_lowers: list[float],
    pred_uppers: list[float],
    method_label: str,
) -> None:
    """Draw RUL predictions with uncertainty on *ax*.

    Args:
        ax: A ``matplotlib.axes.Axes`` instance.
        ts_rul_true: Time stamps for reference RUL.
        ruls_true: Reference RUL values.
        ruls_true_lowers: Lower 95 % CI of reference RUL.
        ruls_true_uppers: Upper 95 % CI of reference RUL.
        ts_pred: Time stamps for model predictions.
        pred_means: Predicted RUL means.
        pred_lowers: Lower 95 % CI of predicted RUL.
        pred_uppers: Upper 95 % CI of predicted RUL.
        method_label: Human-readable method name for the legend.
    """
    import matplotlib.pyplot as plt

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    ax.plot(ts_rul_true, ruls_true, label="True RUL", color=colors[0])
    ax.fill_between(
        ts_rul_true,
        ruls_true_lowers,
        ruls_true_uppers,
        color=colors[0],
        alpha=0.2,
        label="True RUL 95% CI",
    )
    ax.plot(
        ts_pred,
        pred_means,
        label=f"Predicted RUL ({method_label})",
        color=colors[1],
        marker="o",
    )
    ax.fill_between(
        ts_pred,
        pred_lowers,
        pred_uppers,
        color=colors[1],
        alpha=0.2,
        label="Predicted RUL 95% CI",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("RUL [s]")
    ax.set_title(f"{method_label} RUL Predictions with Uncertainty")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    ax.grid(True, alpha=0.3)


def populate_crps_ax(
    ax: Any,
    ts_eval: list[float],
    crps_values: list[float],
    method_label: str,
) -> None:
    """Draw a CRPS-over-time curve on *ax*.

    Args:
        ax: A ``matplotlib.axes.Axes`` instance.
        ts_eval: Time stamps at evaluation points.
        crps_values: CRPS values at each evaluation point.
        method_label: Human-readable method name for the legend.
    """
    ax.plot(ts_eval, crps_values, marker="o", linewidth=2, label=method_label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("CRPS [s]")
    ax.set_title(f"CRPS Over Time — {method_label} vs Simulator")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_rul(
    ts_rul_true: np.ndarray,
    ruls_true: np.ndarray,
    ruls_true_lowers: list[float],
    ruls_true_uppers: list[float],
    ts_pred: list[float],
    pred_means: list[float],
    pred_lowers: list[float],
    pred_uppers: list[float],
    method_label: str,
    output_dir: Path,
) -> None:
    """Generate and save RUL prediction plots."""
    import matplotlib.pyplot as plt

    # Point prediction plot
    fig0, ax0 = plt.subplots(figsize=(10, 6))
    ax0.plot(ts_rul_true, ruls_true, label="True RUL")
    ax0.plot(ts_pred, pred_means, label=f"Predicted RUL ({method_label})", marker="o")
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("RUL [s]")
    ax0.set_title(f"{method_label} RUL Mean Predictions")
    ax0.set_ylim(bottom=0.0)
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    fig0.savefig(output_dir / "rul_point_prediction.png", dpi=150, bbox_inches="tight")
    plt.close(fig0)

    # Uncertainty plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    populate_rul_ax(
        ax1,
        ts_rul_true,
        ruls_true,
        ruls_true_lowers,
        ruls_true_uppers,
        ts_pred,
        pred_means,
        pred_lowers,
        pred_uppers,
        method_label,
    )
    fig1.savefig(
        output_dir / "rul_uncertainty_prediction.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig1)
    print(f"RUL plots saved to {output_dir}")


def _plot_crps(
    ts_eval: list[float],
    crps_values: list[float],
    method_label: str,
    output_dir: Path,
) -> None:
    """Generate and save CRPS over time plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    populate_crps_ax(ax, ts_eval, crps_values, method_label)
    fig.savefig(output_dir / "crps_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"CRPS plot saved to {output_dir / 'crps_over_time.png'}")


def _compute_reference_ci(
    ref_t_eods: np.ndarray,
    n_eval_points: int,
) -> tuple[list[float], list[float]]:
    """Compute 95% CI from reference RUL distributions."""
    lowers = []
    uppers = []
    for k in range(n_eval_points):
        t_eods = ref_t_eods[k]
        lowers.append(float(np.percentile(t_eods, 2.5)))
        uppers.append(float(np.percentile(t_eods, 97.5)))
    return lowers, uppers


# ===================================================================
# generate_data
# ===================================================================


def generate_data(
    *,
    n_histories_train: int | None = None,
    n_histories_val: int | None = None,
    n_test_cases: int = 1,
    n_simu: int = 500,
    n_intermediate_socs: int = 200,
    seed_train: int | None = None,
    seed_val: int | None = None,
    seed_test: int | None = None,
    output_dir: str = "data",
) -> None:
    """Generate train, validation, and test datasets.

    Args:
        n_histories_train: Number of training discharge histories.
        n_histories_val: Number of validation discharge histories.
        n_test_cases: Number of independent test cases.
        n_simu: Stochastic simulations per intermediate SoC for test reference RUL.
        n_intermediate_socs: Number of intermediate SoC evaluation points per test case.
        seed_train: Random seed for training data.
        seed_val: Random seed for validation data.
        seed_test: Random seed for test data.
        output_dir: Output directory.
    """
    data_params = SHARED_PARAMS["data"]
    sim_params = SHARED_PARAMS["simulation"]

    n_histories_train = (
        data_params["n_histories_train"]
        if n_histories_train is None
        else n_histories_train
    )
    n_histories_val = (
        data_params["n_histories_val"] if n_histories_val is None else n_histories_val
    )
    seed_train = seed_train if seed_train is not None else TRAIN_SEED
    seed_val = seed_val if seed_val is not None else TRAIN_SEED + 1
    seed_test = seed_test if seed_test is not None else TEST_SEED

    output_path = Path(output_dir)
    sim_config = _build_runtime_sim_config(sim_params)
    soc_range = sim_params["soc_range"]

    print(f"Generating {n_histories_train} training histories (seed={seed_train})...")
    train_path = generate_train_data(
        simulator_config=sim_config,
        n_histories=n_histories_train,
        soc_range=soc_range,
        seed=seed_train,
        output_path=output_path / "train.npz",
    )
    print(f"  Saved to {train_path}")

    print(f"Generating {n_histories_val} validation histories (seed={seed_val})...")
    val_path = generate_train_data(
        simulator_config=sim_config,
        n_histories=n_histories_val,
        soc_range=soc_range,
        seed=seed_val,
        output_path=output_path / "val.npz",
    )
    print(f"  Saved to {val_path}")

    print(
        f"Generating {n_test_cases} test case(s) "
        f"(n_simu={n_simu}, n_intermediate_socs={n_intermediate_socs}, "
        f"seed={seed_test})..."
    )
    test_paths = generate_test_data(
        simulator_config=sim_config,
        n_test_cases=n_test_cases,
        n_simu=n_simu,
        n_intermediate_socs=n_intermediate_socs,
        seed=seed_test,
        output_dir=output_path,
    )
    for tp in test_paths:
        print(f"  Saved to {tp}")

    print()
    print("Done!")


# ===================================================================
# train
# ===================================================================


def train(
    method: str,
    *,
    n_epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    patience: int | None = None,
    print_every: int | None = None,
    n_filters: int | None = None,
    kernel_size: int | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    normalize: bool | None = None,
    train_data_path: str = "data/train.npz",
    val_data_path: str = "data/val.npz",
    dropout_rate: float = 0.1,
    n_qubits: int = 6,
    n_pqc_layers: int = 1,
) -> None:
    """Train a model using the specified method.

    Args:
        method: One of ``bayes_cnn``, ``het_cnn``, ``mcd_cnn``, ``qavi_cnn``.
        n_epochs: Maximum number of training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        patience: Early stopping patience.
        print_every: Print progress every N epochs.
        n_filters: Number of convolutional filters.
        kernel_size: Convolution kernel size.
        window_size: Time-window size.
        stride: Stride for time windowing.
        normalize: Whether to normalise targets.
        train_data_path: Path to training data file.
        val_data_path: Path to validation data file.
        dropout_rate: Dropout rate (``mcd_cnn`` only).
        n_qubits: Number of qubits (``qavi_cnn`` only).
        n_pqc_layers: Number of PQC layers (``qavi_cnn`` only).
    """
    _validate_method(method)

    tp = SHARED_PARAMS["training"]
    mp = SHARED_PARAMS["model"]
    dp = SHARED_PARAMS["data"]

    n_epochs = n_epochs if n_epochs is not None else tp["n_epochs"]
    lr = lr if lr is not None else tp["lr"]
    batch_size = batch_size if batch_size is not None else tp["batch_size"]
    patience = patience if patience is not None else tp["patience"]
    print_every = print_every if print_every is not None else tp["print_every"]
    n_filters = n_filters if n_filters is not None else mp["n_filters"]
    kernel_size = kernel_size if kernel_size is not None else mp["kernel_size"]
    window_size = window_size if window_size is not None else dp["window_size"]
    stride = stride if stride is not None else dp["stride"]
    normalize = normalize if normalize is not None else dp["normalize"]

    kwargs = dict(
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        print_every=print_every,
        n_filters=n_filters,
        kernel_size=kernel_size,
        window_size=window_size,
        stride=stride,
        normalize=normalize,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
    )

    dispatch = {
        "bayes_cnn": _train_bayes_cnn,
        "het_cnn": _train_het_cnn,
        "mcd_cnn": _train_mcd_cnn,
        "qavi_cnn": _train_qavi_cnn,
    }
    if method == "mcd_cnn":
        kwargs["dropout_rate"] = dropout_rate
    elif method == "qavi_cnn":
        kwargs["n_qubits"] = n_qubits
        kwargs["n_pqc_layers"] = n_pqc_layers

    dispatch[method](**kwargs)


# ---------------------------------------------------------------------------
# Train: het_cnn
# ---------------------------------------------------------------------------


def _train_het_cnn(
    *,
    n_epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    print_every: int,
    n_filters: int,
    kernel_size: int,
    window_size: int,
    stride: int,
    normalize: bool,
    train_data_path: str,
    val_data_path: str,
) -> None:
    np.random.seed(TRAIN_SEED)
    _root_dir, checkpoint_dir, metadata_dir = get_run_dirs("het_cnn/train", create=True)
    sim_config = _build_sim_config(SHARED_PARAMS["simulation"])

    print("=" * 70)
    print("Heteroscedastic CNN Training on Time-Windowed Battery Data")
    print("=" * 70)

    ds_train, ds_val = _load_datasets(
        train_data_path, val_data_path, window_size, stride, normalize
    )
    dataloader_train, dataloader_val = _create_dataloaders(ds_train, ds_val, batch_size)

    print("Creating heteroscedastic CNN model...")
    rngs = nnx.Rngs(0)
    model = HeteroscedasticCNN1D(
        n_filters=n_filters, kernel_size=kernel_size, rngs=rngs
    )
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Model parameters: {n_params}")
    print()

    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_epochs * (len(ds_train) // batch_size),
        alpha=0.1,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            return nll_loss(model, batch, beta=0.5)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(model, batch):
        return nll_loss(model, batch)

    print("Starting training...")
    print("=" * 70)
    early_stopper = EarlyStopper(patience=patience, min_delta=1e-4)

    best_val_loss, _ = train_loop(
        n_epochs=n_epochs,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        train_batch_fn=lambda batch: train_step(model, optimizer, batch),
        eval_batch_fn=lambda batch: eval_step(model, batch),
        early_stopper=early_stopper,
        print_every=print_every,
        on_train_epoch_start=model.train,
        on_val_epoch_start=model.eval,
    )

    metadata: TrainingMetadata = {
        "method": "het_cnn",
        "simulator_config": sim_config,
        "training_params": TrainingParams(
            window_size=window_size,
            stride=stride,
            n_histories_train=SHARED_PARAMS["data"]["n_histories_train"],
            n_histories_val=SHARED_PARAMS["data"]["n_histories_val"],
            soc_range=list(SHARED_PARAMS["simulation"]["soc_range"]),
        ),
        "model_params": BaseModelParams(n_filters=n_filters, kernel_size=kernel_size),
        "scaling_params": ScalingParams(
            normalize=normalize,
            y_max=ds_train.y_max.item() if normalize else 1.0,
        ),
    }
    _save_checkpoint_and_metadata(model, checkpoint_dir, metadata_dir, metadata)
    print("Done!")


# ---------------------------------------------------------------------------
# Train: mcd_cnn
# ---------------------------------------------------------------------------


def _train_mcd_cnn(
    *,
    n_epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    print_every: int,
    n_filters: int,
    kernel_size: int,
    window_size: int,
    stride: int,
    normalize: bool,
    train_data_path: str,
    val_data_path: str,
    dropout_rate: float,
) -> None:
    np.random.seed(TRAIN_SEED)
    _root_dir, checkpoint_dir, metadata_dir = get_run_dirs("mcd_cnn/train", create=True)
    sim_config = _build_sim_config(SHARED_PARAMS["simulation"])

    print("=" * 70)
    print("MC Dropout CNN Training on Time-Windowed Battery Data")
    print("=" * 70)

    ds_train, ds_val = _load_datasets(
        train_data_path, val_data_path, window_size, stride, normalize
    )
    dataloader_train, dataloader_val = _create_dataloaders(ds_train, ds_val, batch_size)

    print("Creating MC Dropout CNN model...")
    rngs = nnx.Rngs(params=0, dropout=1)
    model = MCDCNN1D(
        n_filters=n_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        rngs=rngs,
    )
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Model parameters: {n_params}")
    print()

    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_epochs * (len(ds_train) // batch_size),
        alpha=0.1,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            return nll_loss_mcd(model, batch, beta=0.5)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(model, batch):
        return nll_loss_mcd(model, batch)

    print("Starting training...")
    print("=" * 70)
    early_stopper = EarlyStopper(patience=patience, min_delta=1e-4)

    best_val_loss, _ = train_loop(
        n_epochs=n_epochs,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        train_batch_fn=lambda batch: train_step(model, optimizer, batch),
        eval_batch_fn=lambda batch: eval_step(model, batch),
        early_stopper=early_stopper,
        print_every=print_every,
        on_train_epoch_start=model.train,
        on_val_epoch_start=model.eval,
    )

    metadata: TrainingMetadata = {
        "method": "mcd_cnn",
        "simulator_config": sim_config,
        "training_params": TrainingParams(
            window_size=window_size,
            stride=stride,
            n_histories_train=SHARED_PARAMS["data"]["n_histories_train"],
            n_histories_val=SHARED_PARAMS["data"]["n_histories_val"],
            soc_range=list(SHARED_PARAMS["simulation"]["soc_range"]),
        ),
        "model_params": MCDModelParams(
            n_filters=n_filters,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        ),
        "scaling_params": ScalingParams(
            normalize=normalize,
            y_max=ds_train.y_max.item() if normalize else 1.0,
        ),
    }
    _save_checkpoint_and_metadata(
        model, checkpoint_dir, metadata_dir, metadata, params_only=True
    )
    print("Done!")


# ---------------------------------------------------------------------------
# Train: bayes_cnn
# ---------------------------------------------------------------------------


def _train_bayes_cnn(
    *,
    n_epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    print_every: int,
    n_filters: int,
    kernel_size: int,
    window_size: int,
    stride: int,
    normalize: bool,
    train_data_path: str,
    val_data_path: str,
) -> None:
    np.random.seed(TRAIN_SEED)
    _root_dir, checkpoint_dir, metadata_dir = get_run_dirs(
        "bayes_cnn/train", create=True
    )
    sim_config = _build_sim_config(SHARED_PARAMS["simulation"])

    print("=" * 70)
    print("Bayesian CNN (Flipout) Training on Time-Windowed Battery Data")
    print("=" * 70)

    ds_train, ds_val = _load_datasets(
        train_data_path, val_data_path, window_size, stride, normalize
    )
    dataloader_train, dataloader_val = _create_dataloaders(ds_train, ds_val, batch_size)

    print("Creating Bayesian CNN (Flipout) model...")
    rngs = nnx.Rngs(params=0)
    model = BayesCNN1D(
        conv_cls=FlipoutConv1D,
        n_filters=n_filters,
        kernel_size=kernel_size,
        rngs=rngs,
    )
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Model parameters: {n_params}")
    print()

    n_train = len(ds_train)
    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_epochs * (n_train // batch_size),
        alpha=0.1,
    )
    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)
    base_key = jax.random.PRNGKey(0)

    @nnx.jit
    def train_step(model, optimizer, batch, key):
        def loss_fn(model):
            return elbo_nll_loss(
                model, batch, rngs=nnx.Rngs(params=key), n_train=n_train, beta=0.5
            )

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(model, batch, key):
        return elbo_nll_loss(model, batch, rngs=nnx.Rngs(params=key), n_train=n_train)

    print("Starting training...")
    print("=" * 70)
    early_stopper = EarlyStopper(patience=patience, min_delta=1e-4)
    global_step = 0

    def train_batch_fn(batch: Any) -> None:
        nonlocal global_step
        key = jax.random.fold_in(base_key, global_step)
        train_step(model, optimizer, batch, key)
        global_step += 1

    def eval_batch_fn(batch: Any) -> jax.Array:
        nonlocal global_step
        key = jax.random.fold_in(base_key, global_step)
        loss = eval_step(model, batch, key)
        global_step += 1
        return loss

    best_val_loss, _ = train_loop(
        n_epochs=n_epochs,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        train_batch_fn=train_batch_fn,
        eval_batch_fn=eval_batch_fn,
        early_stopper=early_stopper,
        print_every=print_every,
    )

    metadata: TrainingMetadata = {
        "method": "bayes_cnn",
        "simulator_config": sim_config,
        "training_params": TrainingParams(
            window_size=window_size,
            stride=stride,
            n_histories_train=SHARED_PARAMS["data"]["n_histories_train"],
            n_histories_val=SHARED_PARAMS["data"]["n_histories_val"],
            soc_range=list(SHARED_PARAMS["simulation"]["soc_range"]),
        ),
        "model_params": BaseModelParams(n_filters=n_filters, kernel_size=kernel_size),
        "scaling_params": ScalingParams(
            normalize=normalize,
            y_max=ds_train.y_max.item() if normalize else 1.0,
        ),
    }
    _save_checkpoint_and_metadata(model, checkpoint_dir, metadata_dir, metadata)
    print("Done!")


# ---------------------------------------------------------------------------
# Train: qavi_cnn
# ---------------------------------------------------------------------------


def _train_qavi_cnn(
    *,
    n_epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    print_every: int,
    n_filters: int,
    kernel_size: int,
    window_size: int,
    stride: int,
    normalize: bool,
    train_data_path: str,
    val_data_path: str,
    n_qubits: int,
    n_pqc_layers: int,
) -> None:
    EPS = 1e-7
    BATCH_W = 32
    LR_GEN = lr
    LR_DISC = lr * 0.1
    DISC_HIDDEN = 64
    SEED = TRAIN_SEED

    np.random.seed(SEED)
    _root_dir, checkpoint_dir, metadata_dir = get_run_dirs(
        "qavi_cnn/train", create=True
    )
    sim_config = _build_sim_config(SHARED_PARAMS["simulation"])

    print("=" * 70)
    print("QAVI CNN Training on Time-Windowed Battery Data")
    print("=" * 70)
    print(f"PQC: {n_qubits} qubits, {n_pqc_layers} layers, {n_filters} filters")
    print()

    ds_train, ds_val = _load_datasets(
        train_data_path, val_data_path, window_size, stride, normalize
    )
    dataloader_train, dataloader_val = _create_dataloaders(
        ds_train, ds_val, batch_size, drop_remainder=True
    )

    # Build PQC circuits
    pqc_circuits = [_make_pqc(n_qubits, n_pqc_layers) for _ in range(n_filters)]
    batched_circuits = [jax.vmap(c, in_axes=(None, 0)) for c in pqc_circuits]

    # Initialise components
    rng = jax.random.PRNGKey(SEED)
    q_params_list = []
    for i in range(n_filters):
        rng, k = jax.random.split(rng)
        q_params_list.append(jax.random.normal(k, (n_pqc_layers, n_qubits, 2)) * 0.1)
    q_params_list = tuple(q_params_list)

    pp_list = tuple(
        PostProcessor(n_qubits, rngs=nnx.Rngs(params=SEED + i))
        for i in range(n_filters)
    )
    model = QAVICNN1D(
        n_filters=n_filters,
        kernel_size=kernel_size,
        rngs=nnx.Rngs(params=SEED + n_filters),
    )
    disc_input_dim = window_size + 1
    disc = Discriminator(
        disc_input_dim,
        hidden=DISC_HIDDEN,
        rngs=nnx.Rngs(params=SEED + n_filters + 1),
    )

    # Generator forward (closure over batched_circuits)
    def generator_forward(q_params_list_, pp_list_, z_batch_):
        return _qavi_generator_forward(
            q_params_list_, pp_list_, batched_circuits, z_batch_, n_filters, kernel_size
        )

    # Loss functions
    def disc_loss_fn(disc_, x_batch, y_batch, mu_preds):
        x_flat = x_batch.squeeze(1)
        real_pairs = jnp.concatenate([x_flat, y_batch[:, jnp.newaxis]], axis=-1)
        d_real = disc_(real_pairs)
        loss_real = -jnp.mean(jnp.log(d_real + EPS))

        x_exp = jnp.broadcast_to(
            x_flat[jnp.newaxis, :, :], (mu_preds.shape[0],) + x_flat.shape
        )
        fake_pairs = jnp.concatenate(
            [x_exp, mu_preds[:, :, jnp.newaxis]], axis=-1
        ).reshape(-1, x_flat.shape[-1] + 1)
        d_fake = disc_(fake_pairs).reshape(mu_preds.shape)
        loss_fake = -jnp.mean(jnp.log(1.0 - d_fake + EPS))
        return loss_real + loss_fake

    def gen_loss_fn(
        q_params_list_,
        pp_states_,
        pp_graphdefs_,
        model_state_,
        model_graphdef_,
        disc_,
        z_batch_,
        x_batch,
        y_batch,
    ):
        pp_list_ = tuple(nnx.merge(gd, st) for gd, st in zip(pp_graphdefs_, pp_states_))
        model_ = nnx.merge(model_graphdef_, model_state_)
        kernels, biases = generator_forward(q_params_list_, pp_list_, z_batch_)

        def _single(kernel, bias):
            return model_(x_batch, kernel, bias)

        outputs = jax.vmap(_single)(kernels, biases)
        mu_preds = outputs[:, :, 0]
        var_preds = jnp.clip(outputs[:, :, 1], min=1e-6)

        x_flat = x_batch.squeeze(1)
        x_exp = jnp.broadcast_to(
            x_flat[jnp.newaxis, :, :], (mu_preds.shape[0],) + x_flat.shape
        )
        fake_pairs = jnp.concatenate(
            [x_exp, mu_preds[:, :, jnp.newaxis]], axis=-1
        ).reshape(-1, x_flat.shape[-1] + 1)
        d_fake = disc_(fake_pairs).reshape(mu_preds.shape)
        d_clamped = jnp.clip(d_fake, EPS, 1.0 - EPS)
        logits = jnp.log(d_clamped / (1.0 - d_clamped))
        adv_loss = -jnp.mean(logits)

        y_exp = jnp.broadcast_to(y_batch[jnp.newaxis, :], mu_preds.shape)
        nll = jnp.mean(
            0.5 * jnp.log(var_preds) + 0.5 * jnp.square(y_exp - mu_preds) / var_preds
        )
        return adv_loss + nll

    # Optimizers
    gen_optimizer = optax.adam(LR_GEN)
    disc_optimizer = optax.adam(LR_DISC)

    pp_states_init = tuple(nnx.split(pp)[1] for pp in pp_list)
    _, model_state_init = nnx.split(model)
    gen_opt_state = gen_optimizer.init(
        (q_params_list, pp_states_init, model_state_init)
    )
    _, disc_state_init = nnx.split(disc)
    disc_opt_state = disc_optimizer.init(disc_state_init)

    # JIT'd steps
    @jax.jit
    def disc_step(disc_, disc_opt_state_, x_batch, y_batch, mu_preds):
        graphdef_, state_ = nnx.split(disc_)

        def loss_wrapper(state):
            d = nnx.merge(graphdef_, state)
            return disc_loss_fn(d, x_batch, y_batch, mu_preds)

        loss, grads = jax.value_and_grad(loss_wrapper)(state_)
        updates, new_opt_state = disc_optimizer.update(grads, disc_opt_state_, state_)
        new_state = optax.apply_updates(state_, updates)
        new_disc = nnx.merge(graphdef_, new_state)
        return loss, new_disc, new_opt_state

    @jax.jit
    def gen_step(
        q_params_list_,
        pp_list_,
        model_,
        gen_opt_state_,
        disc_,
        z_batch_,
        x_batch,
        y_batch,
    ):
        pp_splits = [nnx.split(pp) for pp in pp_list_]
        pp_graphdefs_ = tuple(s[0] for s in pp_splits)
        pp_states_ = tuple(s[1] for s in pp_splits)
        model_graphdef_, model_state_ = nnx.split(model_)

        def loss_wrapper(qp, pps, ms):
            return gen_loss_fn(
                qp,
                pps,
                pp_graphdefs_,
                ms,
                model_graphdef_,
                disc_,
                z_batch_,
                x_batch,
                y_batch,
            )

        loss, grads = jax.value_and_grad(loss_wrapper, argnums=(0, 1, 2))(
            q_params_list_, pp_states_, model_state_
        )
        q_grads, pp_grads, m_grads = grads
        gen_params = (q_params_list_, pp_states_, model_state_)
        gen_grads_all = (q_grads, pp_grads, m_grads)
        updates, new_opt_state = gen_optimizer.update(
            gen_grads_all, gen_opt_state_, gen_params
        )
        new_q, new_pp_st, new_m_st = optax.apply_updates(gen_params, updates)
        new_pp_list = tuple(
            nnx.merge(gd, st) for gd, st in zip(pp_graphdefs_, new_pp_st)
        )
        new_model = nnx.merge(model_graphdef_, new_m_st)
        return loss, new_q, new_pp_list, new_model, new_opt_state

    @jax.jit
    def compute_mu_preds(q_params_list_, pp_list_, model_, z_batch_, x_batch):
        kernels, biases = generator_forward(q_params_list_, pp_list_, z_batch_)

        def _single(kernel, bias):
            return model_(x_batch, kernel, bias)

        outputs = jax.vmap(_single)(kernels, biases)
        return outputs[:, :, 0]

    @jax.jit
    def val_step_fn(
        q_params_list_, pp_list_, model_, disc_, z_batch_, x_batch, y_batch
    ):
        pp_splits = [nnx.split(pp) for pp in pp_list_]
        pp_graphdefs_ = tuple(s[0] for s in pp_splits)
        pp_states_ = tuple(s[1] for s in pp_splits)
        model_graphdef_, model_state_ = nnx.split(model_)
        return gen_loss_fn(
            q_params_list_,
            pp_states_,
            pp_graphdefs_,
            model_state_,
            model_graphdef_,
            disc_,
            z_batch_,
            x_batch,
            y_batch,
        )

    # Training loop
    print("Starting QAVI training...")
    print("=" * 70)

    base_key = jax.random.PRNGKey(0)
    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0
    t0 = time.time()

    try:
        for epoch in range(n_epochs):
            train_g_losses = []
            train_d_losses = []

            for batch in dataloader_train:
                x_batch, y_batch = batch
                key = jax.random.fold_in(base_key, global_step)
                k1, k2 = jax.random.split(key)
                z_batch = jax.random.uniform(
                    k1, (BATCH_W,), minval=0.0, maxval=2.0 * jnp.pi
                )

                mu_preds = compute_mu_preds(
                    q_params_list, pp_list, model, z_batch, x_batch
                )
                d_loss, disc, disc_opt_state = disc_step(
                    disc, disc_opt_state, x_batch, y_batch, mu_preds
                )

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

            val_losses = []
            for batch in dataloader_val:
                x_batch, y_batch = batch
                key = jax.random.fold_in(base_key, global_step)
                z_batch = jax.random.uniform(
                    key, (BATCH_W,), minval=0.0, maxval=2.0 * jnp.pi
                )
                v_loss = val_step_fn(
                    q_params_list, pp_list, model, disc, z_batch, x_batch, y_batch
                )
                val_losses.append(float(v_loss))
                global_step += 1

            mean_val = np.mean(val_losses)

            if (epoch + 1) % print_every == 0 or epoch == 0:
                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch + 1:3d}/{n_epochs} | "
                    f"G_loss: {mean_g:.4f} | D_loss: {mean_d:.4f} | "
                    f"Val: {mean_val:.4f} | [{elapsed:.1f}s]"
                )

            if mean_val < best_val_loss - 1e-4:
                best_val_loss = mean_val
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch + 1}")

    elapsed = time.time() - t0
    print("=" * 70)
    print(f"Training complete in {elapsed:.1f}s! Best val loss: {best_val_loss:.6f}")
    print()

    # Save checkpoint
    print("Saving checkpoint...")
    ckpt_base = ocp.test_utils.erase_and_create_empty(checkpoint_dir)

    np.savez(
        ckpt_base / "pqc_params.npz",
        **{f"filter_{i}": np.array(q_params_list[i]) for i in range(n_filters)},
    )

    pp_states_save = [nnx.state(pp, nnx.Param) for pp in pp_list]
    with open(ckpt_base / "pp_states.pkl", "wb") as f:
        pickle.dump(pp_states_save, f)

    checkpointer = ocp.StandardCheckpointer()
    _, model_state_final = nnx.split(model)
    checkpointer.save(ckpt_base / "model_state", model_state_final)
    time.sleep(0.5)
    print(f"Checkpoint saved to {checkpoint_dir}")

    metadata: QAVITrainingMetadata = {
        "method": "qavi_cnn",
        "simulator_config": sim_config,
        "training_params": QAVITrainingParams(
            window_size=window_size,
            stride=stride,
            n_histories_train=SHARED_PARAMS["data"]["n_histories_train"],
            n_histories_val=SHARED_PARAMS["data"]["n_histories_val"],
            soc_range=list(SHARED_PARAMS["simulation"]["soc_range"]),
            batch_w=BATCH_W,
        ),
        "model_params": BaseModelParams(n_filters=n_filters, kernel_size=kernel_size),
        "pqc_params": PQCParams(n_qubits=n_qubits, n_pqc_layers=n_pqc_layers),
        "scaling_params": ScalingParams(
            normalize=normalize,
            y_max=ds_train.y_max.item() if normalize else 1.0,
        ),
    }
    save_metadata(metadata_dir, metadata)
    print(f"Metadata saved to {metadata_dir}")
    print()
    print("Done!")


# ===================================================================
# test
# ===================================================================


def test(
    method: str,
    *,
    test_data_path: str = "data/test_case_0.npz",
    n_samples: int = 500,
    output_dir: str | None = None,
) -> None:
    """Evaluate a trained model: RUL prediction + CRPS.

    Args:
        method: One of ``bayes_cnn``, ``het_cnn``, ``mcd_cnn``, ``qavi_cnn``.
        test_data_path: Path to the test-case ``.npz`` file.
        n_samples: Number of forward passes / weight samples for uncertainty.
        output_dir: Directory for output plots. Defaults to ``saved/<method>/test``.
    """
    _validate_method(method)

    if output_dir is None:
        output_dir = f"saved/{method}/test"

    dispatch = {
        "bayes_cnn": _test_bayes_cnn,
        "het_cnn": _test_het_cnn,
        "mcd_cnn": _test_mcd_cnn,
        "qavi_cnn": _test_qavi_cnn,
    }
    dispatch[method](
        test_data_path=test_data_path,
        n_samples=n_samples,
        output_dir=output_dir,
    )


_PREDICT_DISPATCH: dict[str, Any] = {
    "bayes_cnn": lambda **kw: _predict_bayes_cnn(**kw),
    "het_cnn": lambda **kw: _predict_het_cnn(**kw),
    "mcd_cnn": lambda **kw: _predict_mcd_cnn(**kw),
    "qavi_cnn": lambda **kw: _predict_qavi_cnn(**kw),
}


def compare(
    methods: Sequence[str] | None = None,
    *,
    test_data_path: str = "data/test_case_0.npz",
    n_samples: int = 500,
    output_dir: str | None = None,
) -> Figure:
    """Compare multiple methods on the same test case.

    Produces a figure with one RUL subplot per method and a final subplot
    overlaying the CRPS curves of all methods.

    Args:
        methods: Methods to compare. Defaults to all methods in ``METHODS``.
        test_data_path: Path to the test-case ``.npz`` file.
        n_samples: Number of forward passes / weight samples for uncertainty.
        output_dir: Directory for the output plot. Defaults to
            ``saved/compare``.

    Returns:
        The ``matplotlib.figure.Figure`` containing the comparison subplots.
    """
    import matplotlib.pyplot as plt

    if methods is None:
        methods = list(METHODS)
    for m in methods:
        _validate_method(m)

    if output_dir is None:
        output_dir = "saved/compare"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[TestResult] = []
    for method in methods:
        try:
            result = _PREDICT_DISPATCH[method](
                test_data_path=test_data_path, n_samples=n_samples
            )
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Skipping {method}: {exc}",
                stacklevel=2,
            )

    if not results:
        raise RuntimeError("No methods produced results. Nothing to compare.")

    n_methods = len(results)
    fig, axes = plt.subplots(
        1,
        n_methods + 1,
        figsize=(7 * (n_methods + 1), 6),
    )
    if n_methods + 1 == 1:
        axes = [axes]

    for i, result in enumerate(results):
        populate_rul_ax(
            axes[i],
            result.ts_rul_true,
            result.ruls_true,
            result.ruls_true_lowers,
            result.ruls_true_uppers,
            result.ts_pred,
            result.pred_means,
            result.pred_lowers,
            result.pred_uppers,
            result.method_label,
        )

    crps_ax = axes[-1]
    for result in results:
        populate_crps_ax(
            crps_ax, result.ts_eval, result.crps_values, result.method_label
        )
    crps_ax.set_title("CRPS Over Time — All Methods")

    fig.tight_layout()
    fig.savefig(out_path / "compare.png", dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {out_path / 'compare.png'}")
    return fig


# ---------------------------------------------------------------------------
# Test: het_cnn
# ---------------------------------------------------------------------------


def _predict_het_cnn(*, test_data_path: str, n_samples: int) -> TestResult:
    """Run het_cnn predictions and return a ``TestResult``."""
    np.random.seed(TEST_SEED)
    root_dir, _, metadata_dir = get_run_dirs("het_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"

    metadata = load_metadata(metadata_dir)

    print("=" * 70)
    print("Heteroscedastic CNN — RUL + CRPS Evaluation")
    print("=" * 70)

    print("Loading trained model...")
    model = restore_model_from_checkpoint(
        ckpt_dir / "trained_state",
        lambda: HeteroscedasticCNN1D(**metadata["model_params"], rngs=nnx.Rngs(0)),
    )
    model.eval()
    print("Model loaded successfully.")
    print()

    test_data = np.load(test_data_path)
    discharge_voltage = test_data["voltage"]
    t_eod = float(test_data["t_eod"])
    dt = float(test_data["dt"])
    eval_indices = test_data["eval_indices"]
    ref_t_eods = test_data["ref_t_eods"]
    N_t = len(discharge_voltage)

    window_size = metadata["training_params"]["window_size"]
    stride = metadata["training_params"]["stride"]
    y_max_train = metadata["scaling_params"]["y_max"]

    # --- RUL prediction ---
    print("Computing RUL predictions...")
    ts_pred: list[float] = []
    pred_means: list[float] = []
    pred_lowers: list[float] = []
    pred_uppers: list[float] = []

    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        ts_pred.append(end * dt)
        X = discharge_voltage[start:end].reshape(1, -1)
        pred = model(jnp.expand_dims(X, 0))[0]
        mu = float(pred[0]) * y_max_train
        var = float(pred[1]) * y_max_train**2
        std = np.sqrt(max(var, 1e-12))
        pred_means.append(mu)
        pred_lowers.append(mu - 1.96 * std)
        pred_uppers.append(mu + 1.96 * std)

    N_INTERMEDIATE_SOCs = len(eval_indices)
    ruls_true_lowers, ruls_true_uppers = _compute_reference_ci(
        ref_t_eods, N_INTERMEDIATE_SOCs
    )
    ts_rul_true = (eval_indices.astype(float) * dt).tolist()
    ruls_true = (t_eod - eval_indices.astype(float) * dt).tolist()

    # --- CRPS ---
    valid_mask = eval_indices >= window_size
    crps_eval_indices = eval_indices[valid_mask]
    crps_ref_t_eods = ref_t_eods[valid_mask]

    print("Computing CRPS at each evaluation point...")
    ts_eval: list[float] = []
    crps_values: list[float] = []

    for k, idx in enumerate(crps_eval_indices):
        t = idx * dt
        start = idx - window_size
        ts_eval.append(t)
        window = discharge_voltage[start:idx].reshape(1, -1)
        pred = model(jnp.expand_dims(window, 0))[0]
        mu = float(pred[0]) * y_max_train
        var = float(pred[1]) * y_max_train**2
        std = np.sqrt(max(var, 1e-12))
        pred_samples = np.clip(np.random.normal(mu, std, size=n_samples), 0, None)

        ref_samples = crps_ref_t_eods[k]
        all_samples = np.concatenate([pred_samples, ref_samples])
        x_grid = jnp.linspace(0, float(all_samples.max()) * 1.1, 500)
        crps_val = float(crps(jnp.array(pred_samples), jnp.array(ref_samples), x_grid))
        crps_values.append(crps_val)
        print(
            f"  [{k + 1:2d}/{len(crps_eval_indices)}] t={t:7.1f}s | "
            f"mu={mu:7.1f} | std={std:6.1f} | CRPS={crps_val:.3f}"
        )

    return TestResult(
        method_label=METHOD_LABELS["het_cnn"],
        ts_rul_true=ts_rul_true,
        ruls_true=ruls_true,
        ruls_true_lowers=ruls_true_lowers,
        ruls_true_uppers=ruls_true_uppers,
        ts_pred=ts_pred,
        pred_means=pred_means,
        pred_lowers=pred_lowers,
        pred_uppers=pred_uppers,
        ts_eval=ts_eval,
        crps_values=crps_values,
    )


def _test_het_cnn(*, test_data_path: str, n_samples: int, output_dir: str) -> None:
    result = _predict_het_cnn(test_data_path=test_data_path, n_samples=n_samples)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _plot_rul(
        result.ts_rul_true,
        result.ruls_true,
        result.ruls_true_lowers,
        result.ruls_true_uppers,
        result.ts_pred,
        result.pred_means,
        result.pred_lowers,
        result.pred_uppers,
        result.method_label,
        out_path,
    )
    _plot_crps(result.ts_eval, result.crps_values, result.method_label, out_path)
    print()
    print("Done!")


# ---------------------------------------------------------------------------
# Test: mcd_cnn
# ---------------------------------------------------------------------------


def _predict_mcd_cnn(*, test_data_path: str, n_samples: int) -> TestResult:
    """Run mcd_cnn predictions and return a ``TestResult``."""
    np.random.seed(TEST_SEED)
    root_dir, _, metadata_dir = get_run_dirs("mcd_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"

    metadata = load_metadata(metadata_dir)

    print("=" * 70)
    print("MC Dropout CNN — RUL + CRPS Evaluation")
    print("=" * 70)
    print(f"Number of MC forward passes: {n_samples}")
    print()

    print("Loading trained model...")
    model = MCDCNN1D(**metadata["model_params"], rngs=nnx.Rngs(params=0, dropout=1))
    restore_model_state(ckpt_dir / "trained_state", model)
    model.train()
    rng_dropout = nnx.Rngs(dropout=42)
    print("Model loaded successfully (train mode for MC Dropout).")
    print()

    test_data = np.load(test_data_path)
    discharge_voltage = test_data["voltage"]
    t_eod = float(test_data["t_eod"])
    dt = float(test_data["dt"])
    eval_indices = test_data["eval_indices"]
    ref_t_eods = test_data["ref_t_eods"]
    N_t = len(discharge_voltage)

    window_size = metadata["training_params"]["window_size"]
    stride = metadata["training_params"]["stride"]
    y_max_train = metadata["scaling_params"]["y_max"]

    # --- RUL prediction ---
    print("Computing RUL predictions...")
    ts_pred: list[float] = []
    pred_means: list[float] = []
    pred_lowers: list[float] = []
    pred_uppers: list[float] = []

    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        ts_pred.append(end * dt)
        X = discharge_voltage[start:end].reshape(1, -1)
        x_input = jnp.expand_dims(X, 0)

        mu_samples = []
        full_samples = []
        for _ in range(n_samples):
            pred = model(x_input, rngs=rng_dropout)[0]
            mu = float(pred[0]) * y_max_train
            var = float(pred[1]) * y_max_train**2
            std = np.sqrt(max(var, 1e-12))
            mu_samples.append(mu)
            full_samples.append(np.clip(np.random.normal(mu, std), 0, None))

        pred_means.append(float(np.mean(mu_samples)))
        pred_lowers.append(float(np.percentile(full_samples, 2.5)))
        pred_uppers.append(float(np.percentile(full_samples, 97.5)))

    N_INTERMEDIATE_SOCs = len(eval_indices)
    ruls_true_lowers, ruls_true_uppers = _compute_reference_ci(
        ref_t_eods, N_INTERMEDIATE_SOCs
    )
    ts_rul_true = (eval_indices.astype(float) * dt).tolist()
    ruls_true = (t_eod - eval_indices.astype(float) * dt).tolist()

    # --- CRPS ---
    valid_mask = eval_indices >= window_size
    crps_eval_indices = eval_indices[valid_mask]
    crps_ref_t_eods = ref_t_eods[valid_mask]

    print("Computing CRPS at each evaluation point...")
    ts_eval: list[float] = []
    crps_values: list[float] = []

    for k, idx in enumerate(crps_eval_indices):
        t = idx * dt
        start = idx - window_size
        ts_eval.append(t)
        window = discharge_voltage[start:idx].reshape(1, -1)
        x_input = jnp.expand_dims(window, 0)

        pred_samples = []
        for _ in range(n_samples):
            pred = model(x_input, rngs=rng_dropout)[0]
            mu = float(pred[0]) * y_max_train
            var = float(pred[1]) * y_max_train**2
            std = np.sqrt(max(var, 1e-12))
            pred_samples.append(np.clip(np.random.normal(mu, std), 0, None))
        pred_samples_arr = np.array(pred_samples)

        ref_samples = crps_ref_t_eods[k]
        all_samples = np.concatenate([pred_samples_arr, ref_samples])
        x_grid = jnp.linspace(0, float(all_samples.max()) * 1.1, 500)
        crps_val = float(
            crps(jnp.array(pred_samples_arr), jnp.array(ref_samples), x_grid)
        )
        crps_values.append(crps_val)
        print(
            f"  [{k + 1:2d}/{len(crps_eval_indices)}] t={t:7.1f}s | "
            f"mu={np.mean(pred_samples_arr):7.1f} | "
            f"std={np.std(pred_samples_arr):6.1f} | CRPS={crps_val:.3f}"
        )

    return TestResult(
        method_label=METHOD_LABELS["mcd_cnn"],
        ts_rul_true=ts_rul_true,
        ruls_true=ruls_true,
        ruls_true_lowers=ruls_true_lowers,
        ruls_true_uppers=ruls_true_uppers,
        ts_pred=ts_pred,
        pred_means=pred_means,
        pred_lowers=pred_lowers,
        pred_uppers=pred_uppers,
        ts_eval=ts_eval,
        crps_values=crps_values,
    )


def _test_mcd_cnn(*, test_data_path: str, n_samples: int, output_dir: str) -> None:
    result = _predict_mcd_cnn(test_data_path=test_data_path, n_samples=n_samples)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _plot_rul(
        result.ts_rul_true,
        result.ruls_true,
        result.ruls_true_lowers,
        result.ruls_true_uppers,
        result.ts_pred,
        result.pred_means,
        result.pred_lowers,
        result.pred_uppers,
        result.method_label,
        out_path,
    )
    _plot_crps(result.ts_eval, result.crps_values, result.method_label, out_path)
    print()
    print("Done!")


# ---------------------------------------------------------------------------
# Test: bayes_cnn
# ---------------------------------------------------------------------------


def _predict_bayes_cnn(*, test_data_path: str, n_samples: int) -> TestResult:
    """Run bayes_cnn predictions and return a ``TestResult``."""
    np.random.seed(TEST_SEED)
    root_dir, _, metadata_dir = get_run_dirs("bayes_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"

    metadata = load_metadata(metadata_dir)

    print("=" * 70)
    print("Bayesian CNN (Flipout) — RUL + CRPS Evaluation")
    print("=" * 70)
    print(f"Number of weight samples: {n_samples}")
    print()

    print("Loading trained model...")
    model = restore_model_from_checkpoint(
        ckpt_dir / "trained_state",
        lambda: BayesCNN1D(
            conv_cls=FlipoutConv1D, **metadata["model_params"], rngs=nnx.Rngs(0)
        ),
    )
    print("Model loaded successfully.")
    print()

    test_data = np.load(test_data_path)
    discharge_voltage = test_data["voltage"]
    t_eod = float(test_data["t_eod"])
    dt = float(test_data["dt"])
    eval_indices = test_data["eval_indices"]
    ref_t_eods = test_data["ref_t_eods"]
    N_t = len(discharge_voltage)

    window_size = metadata["training_params"]["window_size"]
    stride = metadata["training_params"]["stride"]
    y_max_train = metadata["scaling_params"]["y_max"]
    base_key = jax.random.PRNGKey(42)

    # --- RUL prediction ---
    print("Computing RUL predictions...")
    ts_pred: list[float] = []
    pred_means: list[float] = []
    pred_lowers: list[float] = []
    pred_uppers: list[float] = []

    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        ts_pred.append(end * dt)
        X = discharge_voltage[start:end].reshape(1, -1)
        x_input = jnp.expand_dims(X, 0)

        mu_samples = []
        full_samples = []
        for i in range(n_samples):
            key = jax.random.fold_in(base_key, i)
            pred = model(x_input, rngs=nnx.Rngs(params=key))[0]
            mu = float(pred[0]) * y_max_train
            var = float(pred[1]) * y_max_train**2
            std = np.sqrt(max(var, 1e-12))
            mu_samples.append(mu)
            full_samples.append(np.clip(np.random.normal(mu, std), 0, None))

        pred_means.append(float(np.mean(mu_samples)))
        pred_lowers.append(float(np.percentile(full_samples, 2.5)))
        pred_uppers.append(float(np.percentile(full_samples, 97.5)))

    N_INTERMEDIATE_SOCs = len(eval_indices)
    ruls_true_lowers, ruls_true_uppers = _compute_reference_ci(
        ref_t_eods, N_INTERMEDIATE_SOCs
    )
    ts_rul_true = (eval_indices.astype(float) * dt).tolist()
    ruls_true = (t_eod - eval_indices.astype(float) * dt).tolist()

    # --- CRPS ---
    valid_mask = eval_indices >= window_size
    crps_eval_indices = eval_indices[valid_mask]
    crps_ref_t_eods = ref_t_eods[valid_mask]

    print("Computing CRPS at each evaluation point...")
    ts_eval: list[float] = []
    crps_values: list[float] = []

    for k, idx in enumerate(crps_eval_indices):
        t = idx * dt
        start = idx - window_size
        ts_eval.append(t)
        window = discharge_voltage[start:idx].reshape(1, -1)
        x_input = jnp.expand_dims(window, 0)

        pred_samples = []
        for i in range(n_samples):
            key = jax.random.fold_in(base_key, k * n_samples + i)
            pred = model(x_input, rngs=nnx.Rngs(params=key))[0]
            mu = float(pred[0]) * y_max_train
            var = float(pred[1]) * y_max_train**2
            std = np.sqrt(max(var, 1e-12))
            pred_samples.append(np.clip(np.random.normal(mu, std), 0, None))
        pred_samples_arr = np.array(pred_samples)

        ref_samples = crps_ref_t_eods[k]
        all_samples = np.concatenate([pred_samples_arr, ref_samples])
        x_grid = jnp.linspace(0, float(all_samples.max()) * 1.1, 500)
        crps_val = float(
            crps(jnp.array(pred_samples_arr), jnp.array(ref_samples), x_grid)
        )
        crps_values.append(crps_val)
        print(
            f"  [{k + 1:2d}/{len(crps_eval_indices)}] t={t:7.1f}s | "
            f"mu={np.mean(pred_samples_arr):7.1f} | "
            f"std={np.std(pred_samples_arr):6.1f} | CRPS={crps_val:.3f}"
        )

    return TestResult(
        method_label=METHOD_LABELS["bayes_cnn"],
        ts_rul_true=ts_rul_true,
        ruls_true=ruls_true,
        ruls_true_lowers=ruls_true_lowers,
        ruls_true_uppers=ruls_true_uppers,
        ts_pred=ts_pred,
        pred_means=pred_means,
        pred_lowers=pred_lowers,
        pred_uppers=pred_uppers,
        ts_eval=ts_eval,
        crps_values=crps_values,
    )


def _test_bayes_cnn(*, test_data_path: str, n_samples: int, output_dir: str) -> None:
    result = _predict_bayes_cnn(test_data_path=test_data_path, n_samples=n_samples)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _plot_rul(
        result.ts_rul_true,
        result.ruls_true,
        result.ruls_true_lowers,
        result.ruls_true_uppers,
        result.ts_pred,
        result.pred_means,
        result.pred_lowers,
        result.pred_uppers,
        result.method_label,
        out_path,
    )
    _plot_crps(result.ts_eval, result.crps_values, result.method_label, out_path)
    print()
    print("Done!")


# ---------------------------------------------------------------------------
# Test: qavi_cnn
# ---------------------------------------------------------------------------


def _predict_qavi_cnn(*, test_data_path: str, n_samples: int) -> TestResult:
    """Run qavi_cnn predictions and return a ``TestResult``."""
    np.random.seed(TEST_SEED)
    root_dir, _, metadata_dir = get_run_dirs("qavi_cnn/train", create=False)
    ckpt_dir = root_dir / "checkpoints"

    metadata = load_metadata(metadata_dir)

    n_filters = metadata["model_params"]["n_filters"]
    kernel_size = metadata["model_params"]["kernel_size"]
    n_qubits = metadata["pqc_params"]["n_qubits"]
    n_pqc_layers = metadata["pqc_params"]["n_pqc_layers"]

    print("=" * 70)
    print("QAVI CNN — RUL + CRPS Evaluation")
    print("=" * 70)
    print(f"Number of weight samples: {n_samples}")
    print()

    # Build PQC circuits
    pqc_circuits = [_make_pqc(n_qubits, n_pqc_layers) for _ in range(n_filters)]
    batched_circuits = [jax.vmap(c, in_axes=(None, 0)) for c in pqc_circuits]

    def generator_forward(q_params_list_, pp_list_, z_batch_):
        return _qavi_generator_forward(
            q_params_list_, pp_list_, batched_circuits, z_batch_, n_filters, kernel_size
        )

    @jax.jit
    def predict_window(q_params_list_, pp_list_, model_, z_batch_, x_input):
        kernels, biases = generator_forward(q_params_list_, pp_list_, z_batch_)

        def _single(kernel, bias):
            return model_(x_input, kernel, bias)

        outputs = jax.vmap(_single)(kernels, biases)
        return outputs[:, 0, :]

    print("Loading trained QAVI CNN model...")
    q_params_list, pp_list, model = _qavi_load_all_components(ckpt_dir, metadata)
    print("Model loaded successfully.")
    print()

    test_data = np.load(test_data_path)
    discharge_voltage = test_data["voltage"]
    t_eod = float(test_data["t_eod"])
    dt = float(test_data["dt"])
    eval_indices = test_data["eval_indices"]
    ref_t_eods = test_data["ref_t_eods"]
    N_t = len(discharge_voltage)

    window_size = metadata["training_params"]["window_size"]
    stride = metadata["training_params"]["stride"]
    y_max_train = metadata["scaling_params"]["y_max"]
    base_key = jax.random.PRNGKey(42)

    z_all = jnp.stack(
        [
            jax.random.uniform(
                jax.random.fold_in(base_key, i),
                (),
                minval=0.0,
                maxval=2.0 * jnp.pi,
            )
            for i in range(n_samples)
        ]
    )

    # --- RUL prediction ---
    print("Computing RUL predictions...")
    ts_pred: list[float] = []
    pred_means: list[float] = []
    pred_lowers: list[float] = []
    pred_uppers: list[float] = []

    for start in range(0, N_t - window_size, stride):
        end = start + window_size
        ts_pred.append(end * dt)
        X = discharge_voltage[start:end].reshape(1, -1)
        x_input = jnp.expand_dims(X, 0)

        preds = predict_window(q_params_list, pp_list, model, z_all, x_input)
        mus = np.array(preds[:, 0]) * y_max_train
        vars_ = np.array(preds[:, 1]) * y_max_train**2
        stds = np.sqrt(np.clip(vars_, 1e-12, None))
        full_samples = np.clip(np.random.normal(mus, stds), 0, None)

        pred_means.append(float(np.mean(mus)))
        pred_lowers.append(float(np.percentile(full_samples, 2.5)))
        pred_uppers.append(float(np.percentile(full_samples, 97.5)))

    N_INTERMEDIATE_SOCs = len(eval_indices)
    ruls_true_lowers, ruls_true_uppers = _compute_reference_ci(
        ref_t_eods, N_INTERMEDIATE_SOCs
    )
    ts_rul_true = (eval_indices.astype(float) * dt).tolist()
    ruls_true = (t_eod - eval_indices.astype(float) * dt).tolist()

    # --- CRPS ---
    valid_mask = eval_indices >= window_size
    crps_eval_indices = eval_indices[valid_mask]
    crps_ref_t_eods = ref_t_eods[valid_mask]

    print("Computing CRPS at each evaluation point...")
    ts_eval: list[float] = []
    crps_values: list[float] = []

    for k, idx in enumerate(crps_eval_indices):
        t = idx * dt
        start = idx - window_size
        ts_eval.append(t)
        window = discharge_voltage[start:idx].reshape(1, -1)
        x_input = jnp.expand_dims(window, 0)

        z_eval = jnp.stack(
            [
                jax.random.uniform(
                    jax.random.fold_in(base_key, k * n_samples + i),
                    (),
                    minval=0.0,
                    maxval=2.0 * jnp.pi,
                )
                for i in range(n_samples)
            ]
        )
        preds = predict_window(q_params_list, pp_list, model, z_eval, x_input)
        mus = np.array(preds[:, 0]) * y_max_train
        vars_ = np.array(preds[:, 1]) * y_max_train**2
        stds = np.sqrt(np.clip(vars_, 1e-12, None))
        pred_samples = np.clip(np.random.normal(mus, stds), 0, None)

        ref_samples = crps_ref_t_eods[k]
        all_samples = np.concatenate([pred_samples, ref_samples])
        x_grid = jnp.linspace(0, float(all_samples.max()) * 1.1, 500)
        crps_val = float(crps(jnp.array(pred_samples), jnp.array(ref_samples), x_grid))
        crps_values.append(crps_val)
        print(
            f"  [{k + 1:2d}/{len(crps_eval_indices)}] t={t:7.1f}s | "
            f"mu={np.mean(pred_samples):7.1f} | "
            f"std={np.std(pred_samples):6.1f} | CRPS={crps_val:.3f}"
        )

    return TestResult(
        method_label=METHOD_LABELS["qavi_cnn"],
        ts_rul_true=ts_rul_true,
        ruls_true=ruls_true,
        ruls_true_lowers=ruls_true_lowers,
        ruls_true_uppers=ruls_true_uppers,
        ts_pred=ts_pred,
        pred_means=pred_means,
        pred_lowers=pred_lowers,
        pred_uppers=pred_uppers,
        ts_eval=ts_eval,
        crps_values=crps_values,
    )


def _test_qavi_cnn(*, test_data_path: str, n_samples: int, output_dir: str) -> None:
    result = _predict_qavi_cnn(test_data_path=test_data_path, n_samples=n_samples)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _plot_rul(
        result.ts_rul_true,
        result.ruls_true,
        result.ruls_true_lowers,
        result.ruls_true_uppers,
        result.ts_pred,
        result.pred_means,
        result.pred_lowers,
        result.pred_uppers,
        result.method_label,
        out_path,
    )
    _plot_crps(result.ts_eval, result.crps_values, result.method_label, out_path)
    print()
    print("Done!")
