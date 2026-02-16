from __future__ import annotations

from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class SimpleCNN1D(nnx.Module):
    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 5,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Simple 1D CNN for time-series RUL prediction with minimal parameters.

        Architecture: Conv1D -> Activation -> Global Average Pooling -> Dense
        Accepts variable-length input windows.

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

        # Dense layer to output single RUL prediction
        self.dense = nnx.Linear(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the CNN.

        Args:
            x (jax.Array): Input with shape (batch, 1, window_size).
                           Will be transposed to (batch, window_size, 1).
                           Accepts variable-length windows.

        Returns:
            jax.Array: Predicted RUL values with shape (batch,).
        """
        # Transpose from (batch, 1, window_size) to (batch, window_size, 1)
        x = jnp.transpose(x, (0, 2, 1))

        # Conv1D with activation
        x = self.conv(x)
        x = self.act_fn(x)

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=1)

        # Dense layer to single output
        x = self.dense(x)

        # Squeeze last dimension: (batch, 1) -> (batch,)
        return x.squeeze(-1)


class HeteroscedasticCNN1D(nnx.Module):
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
        self.gaussian_block = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the heteroscedastic CNN.

        Args:
            x (jax.Array): Input with shape (batch, 1, window_size).
                           Will be transposed to (batch, window_size, 1).
                           Accepts variable-length windows.

        Returns:
            jax.Array: Concatenated [mu, var_positive] with shape (batch, 2).
        """
        # Transpose from (batch, 1, window_size) to (batch, window_size, 1)
        x = jnp.transpose(x, (0, 2, 1))

        # Conv1D with activation
        x = self.conv(x)
        x = self.act_fn(x)

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=1)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gaussian_block(x)


class MCDCNN1D(nnx.Module):
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

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        # GaussianBlock to output mean and variance
        self.gaussian_block = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        """Forward pass through the MC Dropout CNN.

        Args:
            x (jax.Array): Input with shape (batch, 1, window_size).
                           Will be transposed to (batch, window_size, 1).
                           Accepts variable-length windows.
            rngs (nnx.Rngs, optional): RNGs for dropout sampling. When ``None``,
                the dropout layer uses its internal RNG state (required inside
                ``@nnx.jit``).

        Returns:
            jax.Array: Concatenated [mu, var_positive] with shape (batch, 2).
        """
        # Transpose from (batch, 1, window_size) to (batch, window_size, 1)
        x = jnp.transpose(x, (0, 2, 1))

        # Conv1D with activation and dropout
        x = self.conv(x)
        x = self.act_fn(x)
        x = self.dropout(x) if rngs is None else self.dropout(x, rngs=rngs)

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=1)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gaussian_block(x)


class HeteroscedasticCNN1DV1(nnx.Module):
    def __init__(
        self,
        n_filters: int = 8,
        kernel_size: int = 5,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Heteroscedastic 1D CNN with two conv layers for RUL prediction.

        Architecture: Conv1D -> Act -> Conv1D -> Act -> Global Average Pooling ->
        GaussianBlock. Outputs both mean and variance predictions. Accepts
        variable-length input windows.

        Args:
            n_filters (int, optional): Number of convolutional filters per layer.
                Defaults to 8.
            kernel_size (int, optional): Size of the convolutional kernel.
                Defaults to 5.
            act_fn (nnx.Module, optional): Activation function. Defaults to nnx.gelu.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.act_fn = act_fn

        # First conv layer: (batch, length, 1) -> (batch, L1, n_filters)
        self.conv1 = nnx.Conv(
            in_features=1,
            out_features=n_filters,
            kernel_size=(kernel_size,),
            padding="VALID",
            rngs=rngs,
        )

        # Second conv layer: (batch, L1, n_filters) -> (batch, L2, n_filters)
        self.conv2 = nnx.Conv(
            in_features=n_filters,
            out_features=n_filters,
            kernel_size=(kernel_size,),
            padding="VALID",
            rngs=rngs,
        )

        # GaussianBlock to output mean and variance
        self.gaussian_block = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the two-layer heteroscedastic CNN.

        Args:
            x (jax.Array): Input with shape (batch, 1, window_size).
                           Will be transposed to (batch, window_size, 1).
                           Accepts variable-length windows.

        Returns:
            jax.Array: Concatenated [mu, var_positive] with shape (batch, 2).
        """
        # Transpose from (batch, 1, window_size) to (batch, window_size, 1)
        x = jnp.transpose(x, (0, 2, 1))

        # Conv1D layers with activation
        x = self.act_fn(self.conv1(x))
        x = self.act_fn(self.conv2(x))

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=1)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gaussian_block(x)


class GaussianBlock(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs) -> None:
        self.linear_1 = nnx.Linear(input_dim, output_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(input_dim, output_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        mu = self.linear_1(x)
        var = self.linear_2(x)
        var_positive = nnx.softplus(var)
        return jnp.concat([mu, var_positive], axis=1)


class ResNetBlockV0(nnx.Module):
    def __init__(
        self,
        layer_dim: int,
        act_fn: nnx.Module,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """ResNet block with the same structure as in He et al., 2016 (seminal paper)
        and with identity initialization."""
        self.linear_1 = nnx.Linear(layer_dim, layer_dim, rngs=rngs)
        self.linear_2 = nnx.Linear(layer_dim, layer_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(
            layer_dim, rngs=rngs, scale_init=nnx.initializers.zeros
        )

        self.act_fn = act_fn

    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = self.act_fn(self.linear_1(x))
        x = self.linear_2(x)
        x = self.norm(x)
        x = x + residual  # Residual connection
        x = self.act_fn(x)
        return x


class ResNetBlockV1(nnx.Module):
    def __init__(
        self,
        layer_dim: int,
        dropout_rate: float,
        act_fn: nnx.Module,
        *,
        rngs: nnx.Rngs,
    ):
        """ResNet block with layer normalization with identity initialization and
        dropout on the residual branch."""
        self.linear1 = nnx.Linear(layer_dim, layer_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.linear2 = nnx.Linear(layer_dim, layer_dim, rngs=rngs)
        self.norm = nnx.LayerNorm(
            layer_dim,
            rngs=rngs,
            scale_init=nnx.initializers.zeros,
        )

        self.act_fn = act_fn

    def __call__(self, x, rngs: nnx.Rngs):
        residual = x
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.dropout(x, rngs=rngs)  # apply Dropout inside the branch
        x = self.linear2(x)
        x = self.norm(x)  # Starts as 0 contribution due to init
        x = x + residual
        x = self.act_fn(x)
        return x


class MLPBlockV0(nnx.Module):
    def __init__(
        self, hidden_dim: int, dropout_rate: float, act_fn: nnx.Module, rngs: nnx.Rngs
    ):
        """Linear layer with layer normalization and dropout in between."""
        self.linear1 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

        self.act_fn = act_fn

    def __call__(self, x, rngs: nnx.Rngs):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act_fn(x)
        x = self.dropout(x, rngs=rngs)
        return x


class HNNV0(nnx.Module):
    def __init__(
        self,
        dimensions: Sequence[int],
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Heteroscedastic NN with standard MLP architecture (linear layers and
        activation function).

        Args:
            dimensions (Sequence[int]): The dimensions of the layers. The first
                dimension is the number of input features, while the others are
                the number of neurons of the hidden layers. Therefore,
                `N_hidden_layers = len(dimensions) - 1`.
            act_fn (nnx.Module, optional): The activation function. Defaults to nnx.gelu.
            rngs (nnx.Rngs): RNGs for the flax internal modules.
        """
        self.dim_in = dimensions[0]
        self.n_hid_layers = len(dimensions) - 1
        self.act_fn = act_fn

        self.layers = nnx.List(
            [
                nnx.Linear(d_i, d_j, rngs=rngs)
                for d_i, d_j in zip(dimensions[:-1], dimensions[1:])
            ]
        )

        # Final layer is a Gaussian one (output=[mu, softplus(var)])
        self.layers.append(
            GaussianBlock(input_dim=dimensions[-1], output_dim=1, rngs=rngs)
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.act_fn(layer(x))

        # Gaussian layer is applied w/o act function.
        return self.layers[-1](x)


class HNNV1(nnx.Module):
    def __init__(
        self,
        dim_in: int = 1,
        dim_out: int = 1,
        dim_resnet_layers: int = 50,
        num_resnet_layers: int = 2,
        dim_linear_end: int = 10,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Heteroscedastic NN with ResNet layers.

        Initial and final layers are linear.
        """
        self.act_fn = act_fn

        self.linear_start = nnx.Linear(dim_in, dim_resnet_layers, rngs=rngs)

        self.resnets = nnx.List(
            [
                ResNetBlockV0(dim_resnet_layers, act_fn=act_fn, rngs=rngs)
                for _ in range(num_resnet_layers)
            ]
        )

        self.linear_end = nnx.Linear(dim_resnet_layers, dim_linear_end, rngs=rngs)

        self.gaussian = GaussianBlock(dim_linear_end, dim_out, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: Optional[nnx.Rngs] = None) -> jax.Array:
        x = self.act_fn(self.linear_start(x))

        for resnet in self.resnets:
            # incl. already activation function
            x = resnet(x)

        x = self.act_fn(self.linear_end(x))
        return self.gaussian(x, rngs=rngs)


class MCDNetV0(nnx.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_blocks: int = 3,
        dropout_rate: float = 0.1,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        """Dropout network with MLP blocks.

        The blocks include dropout and layer normalization.
        """
        # Project input up to hidden dimension
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)

        self.act_fn = act_fn

        # Stack ResNet Blocks
        self.blocks = nnx.List(
            [
                MLPBlockV0(hidden_dim, dropout_rate, act_fn=act_fn, rngs=rngs)
                for _ in range(num_blocks)
            ]
        )

        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x, rngs: nnx.Rngs):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act_fn(x)
        # No dropout to avoid dropping important features.

        for block in self.blocks:
            x = block(x, rngs=rngs)

        x = self.linear2(x)
        return x


class MCDNetV1(nnx.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 32,
        output_dim: int = 1,
        num_blocks: int = 3,
        dropout_rate: float = 0.1,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ):
        """Dropout network with ResNet blocks.

        Dropout is in the residual branch of every block.
        """
        # Project input up to hidden dimension
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)

        self.act_fn = act_fn

        # Stack ResNet Blocks with dropout
        self.blocks = [
            ResNetBlockV1(hidden_dim, dropout_rate, act_fn, rngs)
            for _ in range(num_blocks)
        ]

        # Final prediction layer
        self.linear2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)

    def __call__(self, x, rngs: nnx.Rngs):
        x = self.linear1(x)
        x = self.norm1(x)
        x = self.act_fn(x)

        for block in self.blocks:
            x = block(x, rngs=rngs)

        x = self.linear2(x)
        return x


class NNEnsemble(nnx.Module):
    pass


def nll_loss(model: nnx.Module, batch: jax.Array) -> jax.Array:
    """Negative log-liklihood loss, based on a Gaussian predictive distribution of the model.
    It implements Equation (31) in https://doi.org/10.1016/j.ymssp.2023.110796.

    Args:
        model (nnx.Module): Gaussian neural network with 2 outputs (mean and variance).
        batch (jax.Array): batched input data.
        rngs (nnx.Rngs): passed to the forward method of the model.

    Returns:
        jax.Array: loss value for the batch.
    """

    xs, labels = batch
    outputs = model(xs)
    means, variances = outputs[:, 0], outputs[:, 1]
    variances = jnp.clip(variances, min=1e-6)
    losses = 0.5 * jnp.log(variances) + 0.5 * jnp.square(labels - means) / variances

    return jnp.mean(losses)


def nll_loss_mcd(
    model: nnx.Module, batch: jax.Array, rngs: Optional[nnx.Rngs] = None
) -> jax.Array:
    """NLL loss for models that require RNGs at call time (e.g. MC Dropout).

    Same formulation as :func:`nll_loss` but forwards ``rngs`` to the model's
    forward pass so that stochastic layers (dropout) receive fresh random keys.

    Args:
        model (nnx.Module): Gaussian neural network with 2 outputs (mean and variance).
        batch (jax.Array): batched input data.
        rngs (nnx.Rngs, optional): passed to the forward method of the model.
            When ``None``, dropout uses its internal RNG state.

    Returns:
        jax.Array: loss value for the batch.
    """
    xs, labels = batch
    outputs = model(xs) if rngs is None else model(xs, rngs=rngs)
    means, variances = outputs[:, 0], outputs[:, 1]
    variances = jnp.clip(variances, min=1e-6)
    losses = 0.5 * jnp.log(variances) + 0.5 * jnp.square(labels - means) / variances

    return jnp.mean(losses)


def mse_loss(model: nnx.Module, batch: jax.Array) -> jax.Array:
    """Mean squared error loss.

    Args:
        model (nnx.Module): neural network model.
        batch (jax.Array): batched input data.
        rngs (nnx.Rngs): passed to the forward method of the model.
    Returns:
        jax.Array: loss value for the batch.
    """
    xs, labels = batch
    outputs = model(xs)
    losses = jnp.square(outputs - labels)

    return jnp.mean(losses)
