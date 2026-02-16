"""Train a minimal CNN on time-windowed battery discharge data for RUL prediction.

This script demonstrates:
- Training on multiple discharge histories (combined time window sources)
- Validation monitoring during training
- Testing on the initial window of a single discharge history
- Minimal CNN architecture (1 conv layer, 4 filters, no pooling)
"""

import time

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import optax
import orbax.checkpoint as ocp
from _shared import (
    create_battery_and_policy,
    get_run_dirs,
    make_simulator_config,
    write_json,
)
from flax import nnx
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import (
    BatterySimulationTimeWindowSource,
    CombinedTimeWindowSource,
    SimpleCNN1D,
)
from qmodem.train import EarlyStopper


def main():
    """Train a CNN on time-windowed battery data and test on initial window."""
    # Directories
    _root_dir, CHECKPOINT_DIR, METADATA_DIR = get_run_dirs(
        "cnn_time_window", create=True
    )

    # Training parameters
    LR = 1e-3
    N_EPOCHS = 200
    BATCH_SIZE = 32
    PATIENCE = 20
    PRINT_EVERY = 10

    # Data parameters
    N_SIMU_TRAIN = 10  # Number of discharge histories for training
    N_SIMU_VAL = 3  # Number of discharge histories for validation
    WINDOW_SIZE = 48  # ~1/10 of typical history length (~486)
    STRIDE = 24  # 50% overlap between windows

    # Battery simulation parameters
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    SOC_0 = 1.0
    DT = 10.0
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2

    battery, discharge_policy = create_battery_and_policy(CURRENT_AMPLITUDE)

    print("=" * 70)
    print("CNN Training on Time-Windowed Battery Data")
    print("=" * 70)
    print(f"Window size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Training simulations: {N_SIMU_TRAIN}")
    print(f"Validation simulations: {N_SIMU_VAL}")
    print()

    # Create training data: combine multiple simulations
    print("Creating training dataset...")
    train_sources = []
    for i in range(N_SIMU_TRAIN):
        sim_config = make_simulator_config(
            n_simu=1,
            v_cut=V_CUT,
            soc_0=SOC_0,
            dt=DT,
            omega_std=OMEGA_STD,
            eta_std=ETA_STD,
            discharge_policy=discharge_policy,
            battery=battery,
        )
        simulator = les.SimulatorSimple(sim_config)
        source = BatterySimulationTimeWindowSource(
            simulator, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True
        )
        train_sources.append(source)
        print(f"  Simulation {i + 1}/{N_SIMU_TRAIN}: {len(source)} windows")

    ds_train = CombinedTimeWindowSource(train_sources)
    y_max_train = float(jnp.max(jnp.array([s.y_max for s in train_sources])))
    print(f"Total training windows: {len(ds_train)}")
    print(f"Training y_max: {y_max_train:.2f}")
    print()

    # Create validation data
    print("Creating validation dataset...")
    val_sources = []
    for i in range(N_SIMU_VAL):
        sim_config = make_simulator_config(
            n_simu=1,
            v_cut=V_CUT,
            soc_0=SOC_0,
            dt=DT,
            omega_std=OMEGA_STD,
            eta_std=ETA_STD,
            discharge_policy=discharge_policy,
            battery=battery,
        )
        simulator = les.SimulatorSimple(sim_config)
        source = BatterySimulationTimeWindowSource(
            simulator, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True
        )
        val_sources.append(source)
        print(f"  Simulation {i + 1}/{N_SIMU_VAL}: {len(source)} windows")

    ds_val = CombinedTimeWindowSource(val_sources)
    print(f"Total validation windows: {len(ds_val)}")
    print()

    # Create test data: single simulation
    print("Creating test dataset (single simulation)...")
    test_sim_config = make_simulator_config(
        n_simu=1,
        v_cut=V_CUT,
        soc_0=SOC_0,
        dt=DT,
        omega_std=OMEGA_STD,
        eta_std=ETA_STD,
        discharge_policy=discharge_policy,
        battery=battery,
    )
    test_simulator = les.SimulatorSimple(test_sim_config)
    ds_test = BatterySimulationTimeWindowSource(
        test_simulator, window_size=WINDOW_SIZE, stride=STRIDE, normalize=True
    )
    print(f"Test windows: {len(ds_test)}")
    print(f"Test y_max: {ds_test.y_max:.2f} (for unscaling)")
    print()

    # Save training y_max for later unscaling (used by prediction scripts)
    write_json(
        METADATA_DIR / "meta.json",
        {"y_max": y_max_train, "window_size": WINDOW_SIZE},
    )

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
    print("Creating CNN model...")
    rngs = nnx.Rngs(0)
    model = SimpleCNN1D(n_filters=4, kernel_size=5, rngs=rngs)

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
            windows, targets = batch
            predictions = model(windows)
            # Compute MSE directly
            return jnp.mean((predictions - targets) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)  # Updated for Flax 0.11+
        return loss

    @nnx.jit
    def eval_step(model, batch):
        windows, targets = batch
        predictions = model(windows)
        # Compute MSE directly
        return jnp.mean((predictions - targets) ** 2)

    # Training loop
    print("Starting training...")
    print("=" * 70)

    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=1e-4)
    best_val_loss = float("inf")

    for epoch in range(N_EPOCHS):
        # Training
        train_losses = []
        for batch in dataloader_train:
            loss = train_step(model, optimizer, batch)
            train_losses.append(loss)

        train_loss = jnp.mean(jnp.array(train_losses))

        # Validation
        val_losses = []
        for batch in dataloader_val:
            loss = eval_step(model, batch)
            val_losses.append(loss)

        val_loss = jnp.mean(jnp.array(val_losses))

        # Print progress
        if (epoch + 1) % PRINT_EVERY == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:3d}/{N_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f}"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Could save checkpoint here if needed

        # Early stopping
        if early_stopper(val_loss):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("=" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    print()

    # Checkpoint the trained model
    print("Saving checkpoint...")
    ckpt_dir = ocp.test_utils.erase_and_create_empty(CHECKPOINT_DIR)
    checkpointer = ocp.StandardCheckpointer()
    _, model_state = nnx.split(model)
    checkpointer.save(ckpt_dir / "trained_state", model_state)
    time.sleep(0.5)  # Prevent shutdown from breaking checkpointing.
    print(f"Checkpoint saved to {CHECKPOINT_DIR}")
    print()

    # Test on initial window
    print("Testing on initial window of test discharge history...")
    initial_window, initial_target = ds_test[0]

    # Add batch dimension and predict
    initial_window_batch = jnp.expand_dims(initial_window, axis=0)
    prediction = model(initial_window_batch)[0]

    # Unscale prediction and target
    prediction_unscaled = prediction * ds_test.y_max
    target_unscaled = initial_target * ds_test.y_max

    print(f"Initial window index: {len(ds_test) - 1}")
    print(f"Normalized prediction: {prediction:.6f}")
    print(f"Normalized target: {initial_target:.6f}")
    print(f"Unscaled prediction: {prediction_unscaled:.2f}")
    print(f"Unscaled target: {target_unscaled:.2f}")
    print(f"Absolute error: {abs(prediction_unscaled - target_unscaled):.2f}")
    print()

    # Additional test: predict all windows
    print("Evaluating on all test windows...")
    test_predictions = []
    test_targets = []
    for i in range(len(ds_test)):
        window, target = ds_test[i]
        window_batch = jnp.expand_dims(window, axis=0)
        pred = model(window_batch)[0]
        test_predictions.append(pred)
        test_targets.append(target)

    test_predictions = jnp.array(test_predictions)
    test_targets = jnp.array(test_targets)
    test_mse = jnp.mean((test_predictions - test_targets) ** 2)

    print(f"Test MSE (normalized): {test_mse:.6f}")
    print(f"Test RMSE (normalized): {jnp.sqrt(test_mse):.6f}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
