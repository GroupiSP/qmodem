import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx
from grain import DataLoader
from grain.samplers import IndexSampler
from grain.transforms import Batch

from qmodem import GaussianHeteroscedasticMLP, make_battery_data, nll_loss


def main() -> None:
    LR = 1e-2
    N_EPOCHS = 100
    PRINT_EVERY = 10
    N_SIMU_TRAIN_DS = 10
    N_SIMU_TEST_DS = 5
    BATCH_SIZE = 50

    rngs = nnx.Rngs(0)

    # Run iid simulations for training and testing.
    _, ds_train = make_battery_data(N_simu=N_SIMU_TRAIN_DS)
    sim_test, ds_test = make_battery_data(N_simu=N_SIMU_TEST_DS)

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
    model = GaussianHeteroscedasticMLP(
        dimensions=[1, 100, 50, 50, 50, 50, 10], rngs=rngs
    )

    # Define the optimizer.
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=LR), wrt=nnx.Param)

    # Define (jitted) training step and test step functions.
    @nnx.jit
    def train_step(
        model: GaussianHeteroscedasticMLP,
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
        model: GaussianHeteroscedasticMLP, rngs: nnx.Rngs, dataset: tuple[jax.Array]
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

    # Plot RUL
    color = plt.cm.rainbow([0.0, 1.0])
    plt.figure()

    t_eod_mean, t_eod_min, t_eod_max = (
        sim_test.t_eod_stats["mean"],
        sim_test.t_eod_stats["min"],
        sim_test.t_eod_stats["max"],
    )
    plt.plot(
        [0.0, t_eod_mean],
        [t_eod_mean, 0.0],
        color=color[0],
        label="True RUL",
        alpha=0.4,
    )
    plt.fill_between(
        [0.0, t_eod_max],
        [t_eod_max, 0.0],
        [t_eod_min, t_eod_min - t_eod_max],
        alpha=0.2,
        color=color[0],
    )

    model.eval()
    plt.plot(
        model(jnp.array(sim_test.v_mean).reshape(-1, 1), rngs=rngs)[:, 0],
        color=color[1],
        label="Mean Predicted RUL",
        alpha=0.4,
    )

    plt.ylim((0.0, t_eod_max * 1.05))

    plt.legend()
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("RUL")
    plt.show()


if __name__ == "__main__":
    main()
