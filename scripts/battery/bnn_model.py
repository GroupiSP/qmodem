from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from qmodem.module import FlipoutConv1D, GaussianBlock


class Net(nnx.Module):
    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 5,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Bayesian 1D CNN for time-series RUL prediction with uncertainty.

        Architecture: BayesConv1D -> Activation -> Global Average Pooling ->
        GaussianBlock. Bayesian version of :class:`HeteroscedasticCNN1D`,
        trainable with ELBO loss (Bayes by Backprop). Accepts variable-length
        input windows.

        Args:
            n_filters: Number of convolutional filters. Defaults to 4.
            kernel_size: Size of the convolutional kernel. Defaults to 5.
            act_fn: Activation function. Defaults to ``nnx.gelu``.
            rngs: RNGs for the flax internal modules.
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.act_fn = act_fn

        self.conv = FlipoutConv1D(
            in_features=1,
            out_features=n_filters,
            kernel_size=kernel_size,
            padding="VALID",
            rngs=rngs,
        )

        # GaussianBlock to output mean and variance
        self.gauss = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass through the Bayesian CNN.

        Args:
            x: Input with shape ``(batch, window_size, 1)``.
                Accepts variable-length windows.
            rngs: RNGs for weight sampling. The ``params`` stream is used
                to draw a key for the Bayesian convolution layer.

        Returns:
            Concatenated ``[mu, var_positive]`` with shape ``(batch, 2)``.
        """
        # Bayesian Conv1D with activation
        x = self.conv(x, rngs=rngs)
        x = self.act_fn(x)

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=-2)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gauss(x)

    def kl_divergence(self) -> jax.Array:
        """Total KL divergence across all Bayesian layers."""
        return self.conv.kl_divergence()
