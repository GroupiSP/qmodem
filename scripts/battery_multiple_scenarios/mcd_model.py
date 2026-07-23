from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from qmodem.module import GaussianBlock


class Net(nnx.Module):
    # TODO: to test.
    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 5,
        dropout_rate: float = 0.1,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """MC Dropout 1D CNN for time-series RUL prediction with uncertainty.

        Architecture: Conv1D -> Activation -> Dropout -> Global Average Pooling ->
        GaussianBlock. Combines aleatoric uncertainty (GaussianBlock) with epistemic
        uncertainty (MC Dropout). Accepts variable-length input windows.

        Args:
            n_filters (int, optional): Number of convolutional filters. Defaults to 4.
            kernel_size (int, optional): Size of the convolutional kernel. Defaults to 5.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
            act_fn (nnx.Module, optional): Activation function. Defaults to nnx.gelu.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.act_fn = act_fn

        self.conv = nnx.Conv(
            in_features=1,
            out_features=n_filters,
            kernel_size=(kernel_size,),
            padding="VALID",
            rngs=rngs,
        )

        self.dropout = nnx.Dropout(dropout_rate, deterministic=False)

        # GaussianBlock to output mean and variance
        self.gauss = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass through the MC Dropout CNN.

        Args:
            x (jax.Array): Input with shape (batch, window_size, 1).
                           Accepts variable-length windows.
            rngs (nnx.Rngs, optional): RNGs for dropout sampling.
        Returns:
            jax.Array: Concatenated [mu, var_positive] with shape (batch, 2).
        """
        # Conv1D with activation and dropout
        x = self.conv(x)
        x = self.act_fn(x)
        x = self.dropout(x, rngs=rngs)

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=-2)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gauss(x)
