import json

import jax
import lib_eod_simulation as les
import optax
from flax import nnx
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import (
    BatterySimulationSingleTimeSource,
    GaussianHeteroscedasticMLP,
    nll_loss,
)


def quickstart_dataloader(N_simu: int = 1, batch_size: int = 10) -> DataLoader:
    with open("./battery_sim_config.json") as fp:
        sim_config = json.load(fp)

    I_discharge = les.ConstantCurrentDischarge(sim_config["I_const_discharge"])

    sim = les.SimulatorSimple(
        N_simu,
        sim_config["v_cut"],
        sim_config["SoC"],
        I_discharge,
        sim_config["model_config"],
    )

    source = BatterySimulationSingleTimeSource(sim)
    sampler = IndexSampler(num_records=len(source), shuffle=True, seed=0)
    return DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[Batch(batch_size=batch_size)],
        worker_count=0,
    )


def main() -> None:
    LR = 1e-2
    N_EPOCHS = 1_000
    PRINT_EVERY = 50

    rngs = nnx.Rngs(0)

    # Run iid simulations for training and testing.
    dataloader_train = quickstart_dataloader(N_simu=40, batch_size=5)
    # dataloader_test = quickstart_dataloader(N_simu=10)

    # Define the model.
    model = GaussianHeteroscedasticMLP(dimensions=[1, 30, 30, 30, 30], rngs=rngs)

    # Run the model on a batch.
    # batch = next(iter(dataloader_train))
    # print(model(batch[0]))

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
