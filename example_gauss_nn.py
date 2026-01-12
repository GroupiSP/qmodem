import jax
import optax
from flax import nnx
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import GaussianHeteroscedasticMLP, make_battery_datasource, nll_loss


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
