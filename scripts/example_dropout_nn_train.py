import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import lib_eod_simulation as les
import optax
import orbax.checkpoint as ocp
from flax import nnx
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import (
    BATT_CONFIG_PATH,
    MLPV0,
    BatterySimulationSource,
)
from qmodem.utils import mkdir_if_not_existent


def create_model_and_optimizer(lr: float):
    rngs = nnx.Rngs(params=0, dropout=1)
    model = MLPV0(rngs=rngs)

    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    return model, optimizer


def main() -> None:
    # Directories
    ROOT_DIR = Path().cwd() / "saved" / "MLPV0"
    CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
    METADATA_DIR = ROOT_DIR / "metadata"
    # Ensure exist for all directories.
    mkdir_if_not_existent([CHECKPOINT_DIR, METADATA_DIR])

    # Training parameters.
    LR = 1e-3
    N_EPOCHS = 50
    PRINT_EVERY = 10
    BATCH_SIZE = 50
    DO_CHECKPOINT = True

    # Battery simulator parameters.
    N_SIMU_TRAIN_DS = 10
    N_SIMU_TEST_DS = 5
    CURRENT_AMPLITUDE = -2.8 * 0.75
    V_CUT = 2.5
    SOC_0 = 1.0
    DT = 10.0
    OMEGA_STD = 1e-3
    ETA_STD = 1e-2

    # Create battery model.
    with open(BATT_CONFIG_PATH) as fp:
        battery_config = json.load(fp)

    battery = les.BatteryModel(battery_config)

    # Create a current discharge policy.
    discharge_policy = les.ConstantCurrentDischarge(CURRENT_AMPLITUDE)

    # Create the battery simulators (1 for train and 1 for validation).
    simulator_train_config = {
        "N_simu": N_SIMU_TRAIN_DS,
        "v_cut": V_CUT,
        "SoC_0": SOC_0,
        "dt": DT,
        "omega_std": OMEGA_STD,
        "eta_std": ETA_STD,
        "I": discharge_policy,
        "battery": battery,
    }

    simulator_test_config = simulator_train_config.copy()
    simulator_test_config["N_simu"] = N_SIMU_TEST_DS

    sim_train = les.SimulatorSimple(simulator_train_config)
    sim_test = les.SimulatorSimple(simulator_train_config)

    # Use the simulators to create the data sources.
    ds_train = BatterySimulationSource(sim_train, normalize=True)
    ds_test = BatterySimulationSource(sim_test, normalize=True)

    # Prepare the train dataloader.
    sampler_train = IndexSampler(
        num_records=len(ds_train), num_epochs=1, shuffle=True, seed=0
    )
    dataloader_train = DataLoader(
        data_source=ds_train,
        sampler=sampler_train,
        operations=[Batch(batch_size=BATCH_SIZE)],
        worker_count=0,
    )

    # Define model and optimizer.
    model, optimizer = create_model_and_optimizer(lr=LR)

    def loss_fn(model, batch, deterministic):
        # deterministic=False enables dropout during training
        # NNX automatically updates the internal dropout RNG state here!
        xs, labels = batch
        predictions = model(xs, deterministic=deterministic)[:, 0]
        loss = jnp.mean((predictions - labels) ** 2)
        return loss

    # Define (jitted) training step and test step functions.
    @nnx.jit
    def train_step(
        model: MLPV0,
        optimizer: nnx.Optimizer,
        batch: tuple[jax.Array],
    ) -> None:
        """One step of the training (parameter and optimizer state update)."""
        grad_fn = nnx.value_and_grad(loss_fn, argnums=0, has_aux=False)
        loss, grads = grad_fn(model, batch, False)
        optimizer.update(model, grads)  # In-place updates.\

    @nnx.jit
    def eval_step(  # TODO: check
        model: MLPV0, dataset: tuple[jax.Array]
    ) -> jax.Array:
        """Evaluates the model over the entire data-source."""
        return loss_fn(model, batch=dataset, deterministic=True)

    # Train the model.
    for epoch in range(1, N_EPOCHS + 1):
        for batch in dataloader_train:
            train_step(model, optimizer, batch)

        if epoch % PRINT_EVERY == 0:
            train_ds_loss = eval_step(model, ds_train[:])
            test_ds_loss = eval_step(model, ds_test[:])

            print(
                f"Epoch: {epoch:3d}, train loss: {train_ds_loss:.4f}, test loss: {test_ds_loss:.4f}"
            )

    # Save metadata (in this case, the y_max used for scaling).
    metadata = {"y_max": ds_train.y_max}
    with open(METADATA_DIR / "meta.json", "w") as fp:
        json.dump(metadata, fp)

    # Checkpoint the trained model.
    if DO_CHECKPOINT:
        ckpt_dir = ocp.test_utils.erase_and_create_empty(CHECKPOINT_DIR)
        checkpointer = ocp.StandardCheckpointer()

        model_state = nnx.state(model, nnx.Param)
        checkpointer.save(ckpt_dir / "trained_state", model_state)

        time.sleep(0.5)  # prevent shutdown from breaking checkpointing.


if __name__ == "__main__":
    main()
