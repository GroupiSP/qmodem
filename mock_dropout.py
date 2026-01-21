import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx


def main():
    # 1. Generate Synthetic Data (Same as before)
    def get_data(n_samples=100):
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, (n_samples, 1), minval=-3, maxval=3)
        noise = jax.random.normal(key, (n_samples, 1)) * 0.1
        y = jnp.sin(x * 2.0) + noise
        return x, y

    # 2. Define the Model (NNX Style)
    class MCDropoutNet(nnx.Module):
        def __init__(self, rngs: nnx.Rngs):
            # We initialize layers immediately in __init__
            # nnx.Linear automatically grabs the 'params' RNG stream
            self.linear1 = nnx.Linear(1, 64, rngs=rngs)
            self.linear2 = nnx.Linear(64, 64, rngs=rngs)
            self.linear3 = nnx.Linear(64, 1, rngs=rngs)

            # nnx.Dropout automatically grabs the 'dropout' RNG stream
            self.dropout1 = nnx.Dropout(0.1, rngs=rngs)
            self.dropout2 = nnx.Dropout(0.1, rngs=rngs)

        def __call__(self, x, deterministic=True):
            x = self.linear1(x)
            x = nnx.relu(x)
            # We pass the deterministic flag to the dropout layer
            x = self.dropout1(x, deterministic=deterministic)

            x = self.linear2(x)
            x = nnx.relu(x)
            x = self.dropout2(x, deterministic=deterministic)

            x = self.linear3(x)
            return x

    # 3. Training Logic
    # In NNX, we can use nnx.Optimizer to wrap the model and handle updates
    def create_model_and_optimizer():
        # Initialize RNGs. We need 'params' for weights and 'dropout' for the masks.
        rngs = nnx.Rngs(params=42, dropout=42)
        model = MCDropoutNet(rngs=rngs)

        # Create an optimizer that wraps the model
        optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)
        return model, optimizer

    @nnx.jit
    def train_step(model, optimizer, batch_x, batch_y):
        def loss_fn(model):
            # deterministic=False enables dropout during training
            # NNX automatically updates the internal dropout RNG state here!
            predictions = model(batch_x, deterministic=False)
            loss = jnp.mean((predictions - batch_y) ** 2)
            return loss

        # value_and_grad w.r.t the model parameters
        loss, grads = nnx.value_and_grad(loss_fn)(model)

        # Update model parameters in-place
        optimizer.update(model, grads)

        return loss

    # 4. MC Inference Logic
    @nnx.jit
    def predict_step_mc(model, x):
        # deterministic=False ensures we get a random mask.
        # Because 'model' is mutable and passed into nnx.jit,
        # the RNG state inside model.dropout advances automatically every call.
        return model(x, deterministic=False)

    # --- Execution ---

    # Setup
    x_train, y_train = get_data(150)
    model, optimizer = create_model_and_optimizer()

    # Training Loop
    print("Training...")
    for epoch in range(2000):
        loss = train_step(model, optimizer, x_train, y_train)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # --- MC Inference ---
    print("\nRunning MC Dropout Inference...")

    x_test = jnp.linspace(-4, 4, 100).reshape(-1, 1)

    # We want 100 different predictions.
    # Since NNX models are stateful (they hold the RNG key),
    # we can just call the predict function in a loop.
    mc_predictions = []

    for _ in range(100):
        # Every time we call this, the internal dropout RNG advances
        y_pred = predict_step_mc(model, x_test)
        mc_predictions.append(y_pred)

    mc_predictions = jnp.stack(mc_predictions).squeeze()

    # Calculate Statistics
    mean_prediction = jnp.mean(mc_predictions, axis=0)
    std_prediction = jnp.std(mc_predictions, axis=0)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    plt.scatter(x_train, y_train, color="black", s=10, label="Training Data", zorder=5)
    plt.plot(
        x_test, mean_prediction, color="blue", label="Mean Prediction", linewidth=2
    )
    plt.fill_between(
        x_test.flatten(),
        mean_prediction - 2 * std_prediction,
        mean_prediction + 2 * std_prediction,
        color="blue",
        alpha=0.2,
        label="Uncertainty (2 Std)",
    )
    plt.title("Fitting Function with MC Dropout (Flax NNX)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
