"""Train a heteroscedastic CNN on time-windowed battery discharge data for RUL
.. deprecated:: Use the ``qmodem`` CLI instead.  See ``qmodem --help``.

prediction with uncertainty.

This script demonstrates:
- Training on multiple discharge histories (combined time window sources)
- Validation monitoring during training
- Using NLL loss for heteroscedastic predictions (mean + variance)
- Two-layer CNN architecture (1 conv layer, 4 filters of size 5, global average pooling, Gaussian output)
"""

import pickle
import sys
import time
from pathlib import Path

import jax
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import (  # noqa: E402
    SHARED_PARAMS,
    TRAIN_SEED,
    create_battery_and_policy,
    get_run_dirs,
    make_simulator_config,
)
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import (
    BatterySimulationTimeWindowSource,
    HeteroscedasticCNN1D,
    nll_loss,
)
from qmodem.train import EarlyStopper, train_loop


def main():
    """Train a heteroscedastic CNN on time-windowed battery data and test on initial
    window."""
    np.random.seed(TRAIN_SEED)
    # Directories
    _root_dir, CHECKPOINT_DIR, METADATA_DIR = get_run_dirs("het_cnn/train", create=True)

    # Training parameters
    LR = SHARED_PARAMS["training"]["lr"]
    N_EPOCHS = SHARED_PARAMS["training"]["n_epochs"]
    BATCH_SIZE = SHARED_PARAMS["training"]["batch_size"]
    PATIENCE = SHARED_PARAMS["training"]["patience"]
    PRINT_EVERY = SHARED_PARAMS["training"]["print_every"]
    N_FILTERS = SHARED_PARAMS["model"]["n_filters"]
    KERNEL_SIZE = SHARED_PARAMS["model"]["kernel_size"]

    # Data parameters
    N_HISTORIES_TRAIN = SHARED_PARAMS["data"]["n_histories_train"]
    N_HISTORIES_VAL = SHARED_PARAMS["data"]["n_histories_val"]
    WINDOW_SIZE = SHARED_PARAMS["data"]["window_size"]
    STRIDE = SHARED_PARAMS["data"]["stride"]
    NORMALIZE = SHARED_PARAMS["data"]["normalize"]

    # Battery simulation parameters
    CURRENT_AMPLITUDE = SHARED_PARAMS["simulation"]["current_amplitude"]
    V_CUT = SHARED_PARAMS["simulation"]["v_cut"]
    DT = SHARED_PARAMS["simulation"]["dt"]
    OMEGA_STD = SHARED_PARAMS["simulation"]["omega_std"]
    ETA_STD = SHARED_PARAMS["simulation"]["eta_std"]
    SOC_RANGE = SHARED_PARAMS["simulation"]["soc_range"]

    battery, discharge_policy = create_battery_and_policy(CURRENT_AMPLITUDE)

    print("=" * 70)
    print("Heteroscedastic CNN Training on Time-Windowed Battery Data")
    print("=" * 70)
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Training histories: {N_HISTORIES_TRAIN}")
    print(f"Validation histories: {N_HISTORIES_VAL}")
    print(f"SoC₀ range: {SOC_RANGE}")
    print()

    # Base simulator config (N_simu and SoC_0 are set per-history by the data source)
    sim_config = make_simulator_config(
        n_simu=1,
        v_cut=V_CUT,
        soc_0=1.0,
        dt=DT,
        omega_std=OMEGA_STD,
        eta_std=ETA_STD,
        discharge_policy=discharge_policy,
        battery=battery,
    )

    # Create training data: each history starts from a random SoC₀
    print("Loading training dataset...")
    ds_train = BatterySimulationTimeWindowSource.from_file(
        path="data/train.npz",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalize=NORMALIZE,
    )
    print(f"Total training windows: {len(ds_train)}")
    print()

    # Create validation data
    print("Loading validation dataset...")
    ds_val = BatterySimulationTimeWindowSource.from_file(
        path="data/val.npz",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        normalize=NORMALIZE,
    )
    print(f"Total validation windows: {len(ds_val)}")
    print()

    # Create DataLoaders
    sampler_train = IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=True, seed=42
    )
    dataloader_train = DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[Batch(batch_size=BATCH_SIZE)],
        worker_count=0,
    )

    sampler_val = IndexSampler(
        num_records=len(ds_val), num_epochs=1, shuffle=False, seed=0
    )
    dataloader_val = DataLoader(
        data_source=ds_val,
        sampler=sampler_val,
        operations=[Batch(batch_size=BATCH_SIZE)],
        worker_count=0,
    )

    # Create model
    print("Creating heteroscedastic CNN model...")
    rngs = nnx.Rngs(0)
    model = HeteroscedasticCNN1D(
        n_filters=N_FILTERS, kernel_size=KERNEL_SIZE, rngs=rngs
    )

    # Count parameters
    n_params = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Model parameters: {n_params}")
    print()

    # Create optimizer with cosine decay learning rate schedule
    schedule = optax.cosine_decay_schedule(
        init_value=LR,
        decay_steps=N_EPOCHS * (len(ds_train) // BATCH_SIZE),
        alpha=0.1,  # minimum learning rate = 0.1 * lr
    )

    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    # Define training and evaluation functions
    @nnx.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            return nll_loss(model, batch, beta=0.5)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(model, batch):
        """The evaluation loss is just the NLL without the variance weighting.

        The beta weighting is only used during training to encourage the model to learn
        reasonable variances early on.
        """
        return nll_loss(model, batch)

    # Training loop
    print("Starting training...")
    print("=" * 70)

    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=1e-4)

    best_val_loss, _ = train_loop(
        n_epochs=N_EPOCHS,
        dataloader_train=dataloader_train,
        dataloader_val=dataloader_val,
        train_batch_fn=lambda batch: train_step(model, optimizer, batch),
        eval_batch_fn=lambda batch: eval_step(model, batch),
        early_stopper=early_stopper,
        print_every=PRINT_EVERY,
        on_train_epoch_start=model.train,
        on_val_epoch_start=model.eval,
    )

    # Checkpoint the trained model
    print("Saving checkpoint...")
    ckpt_dir = ocp.test_utils.erase_and_create_empty(CHECKPOINT_DIR)
    checkpointer = ocp.StandardCheckpointer()
    _, model_state = nnx.split(model)
    checkpointer.save(ckpt_dir / "trained_state", model_state)
    time.sleep(0.5)  # Prevent shutdown from breaking checkpointing.
    print(f"Checkpoint saved to {CHECKPOINT_DIR}")

    # Save metadata
    metadata = {
        "simulator_config": sim_config,
        "training_params": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "n_histories_train": N_HISTORIES_TRAIN,
            "n_histories_val": N_HISTORIES_VAL,
            "soc_range": SOC_RANGE,
        },
        "model_params": {
            "n_filters": N_FILTERS,
            "kernel_size": KERNEL_SIZE,
        },
        # TODO - y_max should be exposed as a scalar by the datasource, since it's only use is for scaling
        "scaling_params": {
            "normalize": NORMALIZE,
            "y_max": ds_train.y_max.item() if NORMALIZE else 1.0,
        },
    }
    with open(METADATA_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print()
    print(f"Metadata saved to {METADATA_DIR}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()
