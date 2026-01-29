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
    BatterySimulationSource,
    MCDNetV0,
)
from qmodem.train import EarlyStopper
from qmodem.utils import mkdir_if_not_existent


def create_model_and_optimizer(lr_init: float, n_epochs: int, steps_per_epoch: int):
    model = MCDNetV0(rngs=nnx.Rngs(0))

    # Cosine decay schedule
    total_steps = n_epochs * steps_per_epoch
    schedule = optax.cosine_decay_schedule(
        init_value=lr_init,
        decay_steps=total_steps,
        alpha=0.1,  # minimum learning rate = 0.1 * lr
    )

    optimizer = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)
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
    N_SIMU_VAL_DS = 5
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

    simulator_validation_config = simulator_train_config.copy()
    simulator_validation_config["N_simu"] = N_SIMU_VAL_DS

    sim_train = les.SimulatorSimple(simulator_train_config)
    simulator_validation = les.SimulatorSimple(simulator_validation_config)

    # Use the simulators to create the data sources.
    ds_train = BatterySimulationSource(sim_train, normalize=True)
    ds_validation = BatterySimulationSource(simulator_validation, normalize=True)

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
    model, optimizer = create_model_and_optimizer(
        lr_init=LR, n_epochs=N_EPOCHS, steps_per_epoch=len(ds_train) // BATCH_SIZE
    )

    @nnx.vmap(in_axes=(None, 0, 0), out_axes=0)  # Batched over samples and rngs.
    def model_forward_batch(model: MCDNetV0, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return model(x, rngs=rngs)

    def loss_fn(
        model: MCDNetV0, batch: tuple[jax.Array, jax.Array], rngs: nnx.Rngs
    ) -> jax.Array:
        x, y = batch
        predictions = model_forward_batch(model, x, rngs=rngs)
        loss = jnp.mean((predictions[:, 0] - y) ** 2)
        return loss

    # Define (jitted) training step and test step functions.
    @nnx.jit
    def train_step(
        model: MCDNetV0,
        optimizer: nnx.Optimizer,
        batch: tuple[jax.Array],
        rngs: nnx.Rngs,
    ) -> None:
        """One step of the training (parameter and optimizer state update)."""
        grad_fn = nnx.value_and_grad(loss_fn, argnums=0, has_aux=False)
        _, grads = grad_fn(model, batch, rngs)
        optimizer.update(model, grads)

    @nnx.jit
    def eval_step(model: MCDNetV0, dataset: tuple[jax.Array, jax.Array]) -> jax.Array:
        """Evaluates the model over the entire data-source.

        Not vmapped.
        """
        x, y = dataset
        predictions = model(x, rngs=nnx.Rngs(0))
        return jnp.mean((predictions[:, 0] - y) ** 2)

    # Monitor the validation loss for early stopping.
    early_stopper = EarlyStopper(patience=10, min_delta=1e-4)

    # Train the model.
    rng_dropout = nnx.Rngs(1)
    forked_rngs_dropout = rng_dropout.fork(split=BATCH_SIZE)
    forked_rngs_dropout_last = rng_dropout.fork(split=len(ds_train) % BATCH_SIZE)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()  # set model to training mode (for dropout)
        for batch in dataloader_train:
            rngs = (
                forked_rngs_dropout
                if len(batch[0]) == BATCH_SIZE
                else forked_rngs_dropout_last
            )
            train_step(model, optimizer, batch, rngs)

        model.eval()  # set model to eval mode (no dropout)
        val_loss = eval_step(model, ds_validation[:])

        if early_stopper(val_loss):
            break

        if epoch % PRINT_EVERY == 0:
            # Also compute train loss for logging.
            train_ds_loss = eval_step(model, ds_train[:])

            print(
                f"Epoch: {epoch:3d}, train loss: {train_ds_loss:.4f}, validation loss: {val_loss:.4f}"
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
