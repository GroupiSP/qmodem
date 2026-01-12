import json
from pathlib import Path

import jax
import lib_eod_simulation as les
import optax
from flax import nnx
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import (
    BatterySimulationSource,
    GaussianHeteroscedasticMLP,
    nll_loss,
)

SIM_CONFIG_FILE_PATH = Path(__file__).resolve().parent / "battery_sim_config.json"


def make_battery_datasource(N_simu: int = 1) -> BatterySimulationSource:
    """Makes the Grain data source for the battery simulator. Assumes a constant current
    policy.

    Args:
        N_simu (int, optional): Number of MC simulations of the battery discharge. Defaults to 1.

    Returns:
        BatterySimulationSource: the Grain battery data-source.
    """
    with open(SIM_CONFIG_FILE_PATH) as fp:
        sim_config = json.load(fp)

    I_discharge = les.ConstantCurrentDischarge(sim_config["I_const_discharge"])

    sim = les.SimulatorSimple(
        N_simu,
        sim_config["v_cut"],
        sim_config["SoC"],
        I_discharge,
        sim_config["model_config"],
    )

    return BatterySimulationSource(sim)


def main() -> None:
    LR = 1e-2
    N_EPOCHS = 100
    PRINT_EVERY = 5
    N_SIMU_TRAIN_DS = 10
    # N_SIMU_TEST_DS = 5
    BATCH_SIZE = 50

    rngs = nnx.Rngs(0)

    # Run iid simulations for training and testing.
    ds_train = make_battery_datasource(N_simu=N_SIMU_TRAIN_DS)
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
    model = GaussianHeteroscedasticMLP(dimensions=[1, 30, 30, 30, 30], rngs=rngs)

    # Define the optimizer.
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LR), wrt=nnx.Param)

    # Define a (jitted) training step function.
    @nnx.jit
    def train_step(
        model: GaussianHeteroscedasticMLP,
        optimizer: nnx.Optimizer,
        rngs: nnx.Rngs,
        batch: tuple[jax.Array],
    ) -> jax.Array:
        """One step of the training (parameter and optimizer state update)."""
        grad_fn = nnx.value_and_grad(nll_loss, argnums=0, has_aux=False)
        loss, grads = grad_fn(model, batch, rngs)
        optimizer.update(model, grads)  # In-place updates.\
        return loss

    # Train the model.
    model.train()

    for epoch in range(N_EPOCHS):
        for batch in dataloader_train:
            loss = train_step(model, optimizer, rngs, batch)

        if (epoch + 1) % PRINT_EVERY == 0:
            print(f"Epoch: {epoch:3d}, loss: {loss:.4f}")


if __name__ == "__main__":
    main()
