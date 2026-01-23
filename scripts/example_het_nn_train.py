import json
import time
from pathlib import Path

import jax
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
    HeteroscedasticResNet,
    nll_loss,
)


def main() -> None:
    rngs = nnx.Rngs(0)

    # Training parameters.
    LR = 1e-3
    N_EPOCHS = 50
    PRINT_EVERY = 10
    BATCH_SIZE = 50
    DO_CHECKPOINT = False

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
    ds_train = BatterySimulationSource(sim_train)
    ds_test = BatterySimulationSource(sim_test)

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

    # Define the model.
    model = HeteroscedasticResNet(rngs=rngs)

    # Define the optimizer.
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LR), wrt=nnx.Param)

    # Define (jitted) training step and test step functions.
    @nnx.jit
    def train_step(
        model: HeteroscedasticResNet,
        optimizer: nnx.Optimizer,
        rngs: nnx.Rngs,
        batch: tuple[jax.Array],
    ) -> None:
        """One step of the training (parameter and optimizer state update)."""
        grad_fn = nnx.value_and_grad(nll_loss, argnums=0, has_aux=False)
        loss, grads = grad_fn(model, batch, rngs)
        optimizer.update(model, grads)  # In-place updates.\

    @nnx.jit
    def eval_step(
        model: HeteroscedasticResNet, rngs: nnx.Rngs, dataset: tuple[jax.Array]
    ) -> jax.Array:
        """Evaluates the model over the entire data-source."""
        return nll_loss(model, batch=dataset, rngs=rngs)

    # Train the model.
    for epoch in range(1, N_EPOCHS + 1):
        model.train()

        for batch in dataloader_train:
            train_step(model, optimizer, rngs, batch)

        if epoch % PRINT_EVERY == 0:
            model.eval()

            train_ds_loss = eval_step(model, rngs, ds_train[:])
            test_ds_loss = eval_step(model, rngs, ds_test[:])

            print(
                f"Epoch: {epoch:3d}, train loss: {train_ds_loss:.4f}, test loss: {test_ds_loss:.4f}"
            )

    # Checkpoint the trained model.
    if DO_CHECKPOINT:
        ckpt_dir = ocp.test_utils.erase_and_create_empty(
            Path().cwd() / "checkpoints/het_resnet/"
        )
        checkpointer = ocp.StandardCheckpointer()

        _, model_state = nnx.split(model)
        checkpointer.save(ckpt_dir / "trained_state", model_state)

        time.sleep(0.5)  # prevent shutdown to break checkpointing.


if __name__ == "__main__":
    main()
