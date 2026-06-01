import flax.nnx as nnx
import jax
import jax.numpy as jnp

from qmodem.module import GaussianBlock


class Net(nnx.Module):
    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 5,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Heteroscedastic 1D CNN for time-series RUL prediction with uncertainty.

        Architecture: Conv1D -> Activation -> Global Average Pooling -> GaussianBlock
        Outputs both mean and variance predictions. Accepts variable-length input
        windows.

        Args:
            n_filters (int, optional): Number of convolutional filters. Defaults to 4.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 5.
            act_fn (nnx.Module, optional): Activation function. Defaults to nnx.gelu.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.act_fn = act_fn

        self.conv = nnx.Conv(
            in_features=1,
            out_features=n_filters,
            kernel_size=(kernel_size,),
            padding="VALID",
            rngs=rngs,
        )

        # GaussianBlock to output mean and variance
        self.gauss = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the heteroscedastic CNN.

        Args:
            x (jax.Array): Input with shape (batch, window_size, 1).
                           Accepts variable-length windows.

        Returns:
            jax.Array: Concatenated [mu, var_positive] with shape (batch, 2).
        """
        # Conv1D with activation
        x = self.conv(x)
        x = self.act_fn(x)

        # Global Average Pooling: (batch, window_size, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=1)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gauss(x)
