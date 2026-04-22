from __future__ import annotations

from typing import Optional, Protocol, Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class RandomCallModel(Protocol):
    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array: ...


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
        # Bias initialised to -3 so that softplus(-3) ≈ 0.049 at the start of
        # training. This keeps the predicted variance small in early epochs,
        # ensuring the mean head receives a strong gradient signal before the
        # variance has any chance to inflate and suppress it (the canonical NLL
        # "variance collapse" failure mode).
        self.linear_2 = nnx.Linear(
            input_dim,
            output_dim,
            rngs=rngs,
            bias_init=nnx.initializers.constant(-3.0),
        )

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


class StandardBayesConv1D(nnx.Module):
    """Bayesian 1D convolution with shared perturbation (reparameterisation trick).

    Each kernel and bias weight follows q(w) = N(μ, softplus(ρ)²). A single noise draw ε
    is shared across every sample in the batch.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        padding: str = "VALID",
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the variational parameters of the kernel and bias distributions.

        Args:
            in_features: Number of input channels.
            out_features: Number of output channels (filters).
            kernel_size: Spatial size of the 1-D convolution kernel.
            padding: Convolution padding mode (``"VALID"`` or ``"SAME"``).
            rngs: RNGs for parameter initialisation.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding

        k_shape = (kernel_size, in_features, out_features)
        self.kernel_mu = nnx.Param(jax.random.normal(rngs.params(), k_shape) * 0.1)
        self.kernel_rho = nnx.Param(jnp.full(k_shape, -3.0))
        self.bias_mu = nnx.Param(jnp.zeros(out_features))
        self.bias_rho = nnx.Param(jnp.full(out_features, -3.0))

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        """Forward pass: sample one set of weights and convolve the batch.

        Args:
            x: Input with shape ``(batch, length, in_features)``.
            key: JAX PRNG key for weight sampling.

        Returns:
            Convolved output with shape ``(batch, L_out, out_features)``.
        """
        k1, k2 = jax.random.split(key)
        k_sigma = jax.nn.softplus(self.kernel_rho.value)
        b_sigma = jax.nn.softplus(self.bias_rho.value)

        eps_k = jax.random.normal(k1, self.kernel_mu.value.shape)
        eps_b = jax.random.normal(k2, self.bias_mu.value.shape)

        kernel = self.kernel_mu.value + k_sigma * eps_k
        bias = self.bias_mu.value + b_sigma * eps_b

        out = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1,),
            padding=self.padding,
            dimension_numbers=("NHC", "HIO", "NHC"),
        )
        return out + bias

    def kl_divergence(self) -> jax.Array:
        """KL(q ‖ p) with unit-normal prior p = N(0, 1)."""

        def _kl(mu: jax.Array, rho: jax.Array) -> jax.Array:
            sigma = jax.nn.softplus(rho)
            return -0.5 * jnp.sum(1.0 + 2.0 * jnp.log(sigma) - mu**2 - sigma**2)

        return _kl(self.kernel_mu.value, self.kernel_rho.value) + _kl(
            self.bias_mu.value, self.bias_rho.value
        )


class FlipoutConv1D(nnx.Module):
    """Bayesian 1D convolution with Flipout (Wen et al., 2018).

    Each sample gets a pseudo-independent perturbation via per-sample random
    sign vectors on input and output channels::

        y_i = conv(x_i, μ) + b_μ  +  r_i ⊙ [conv(s_i ⊙ x_i, σ ⊙ ε) + σ_b ⊙ ε_b]

    where s_i ∈ {±1}^in and r_i ∈ {±1}^out are Rademacher draws.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        *,
        padding: str = "VALID",
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the variational parameters of the kernel and bias distributions.

        Args:
            in_features: Number of input channels.
            out_features: Number of output channels (filters).
            kernel_size: Spatial size of the 1-D convolution kernel.
            padding: Convolution padding mode (``"VALID"`` or ``"SAME"``).
            rngs: RNGs for parameter initialisation.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding

        k_shape = (kernel_size, in_features, out_features)
        self.kernel_mu = nnx.Param(jax.random.normal(rngs.params(), k_shape) * 0.1)
        self.kernel_rho = nnx.Param(jnp.full(k_shape, -3.0))
        self.bias_mu = nnx.Param(jnp.zeros(out_features))
        self.bias_rho = nnx.Param(jnp.full(out_features, -3.0))

    def __call__(self, x: jax.Array, *, key: jax.Array) -> jax.Array:
        """Forward pass with per-sample sign-flipped perturbations.

        Args:
            x: Input with shape ``(batch, length, in_features)``.
            key: JAX PRNG key for weight and sign sampling.

        Returns:
            Convolved output with shape ``(batch, L_out, out_features)``.
        """
        k1, k2, k3, k4 = jax.random.split(key, 4)
        batch = x.shape[0]
        k_sigma = jax.nn.softplus(self.kernel_rho.value)
        b_sigma = jax.nn.softplus(self.bias_rho.value)

        # Deterministic mean path
        mean_out = (
            jax.lax.conv_general_dilated(
                x,
                self.kernel_mu.value,
                window_strides=(1,),
                padding=self.padding,
                dimension_numbers=("NHC", "HIO", "NHC"),
            )
            + self.bias_mu.value
        )

        # Shared base noise
        eps_k = jax.random.normal(k1, self.kernel_mu.value.shape)
        eps_b = jax.random.normal(k2, self.bias_mu.value.shape)

        # Per-sample sign flips on input/output channels
        s = jax.random.rademacher(k3, (batch, 1, self.in_features)).astype(x.dtype)
        r = jax.random.rademacher(k4, (batch, 1, self.out_features)).astype(x.dtype)

        perturb = jax.lax.conv_general_dilated(
            s * x,
            k_sigma * eps_k,
            window_strides=(1,),
            padding=self.padding,
            dimension_numbers=("NHC", "HIO", "NHC"),
        )
        perturb = r * (perturb + b_sigma * eps_b)
        return mean_out + perturb

    def kl_divergence(self) -> jax.Array:
        """KL(q ‖ p) with unit-normal prior p = N(0, 1)."""

        def _kl(mu: jax.Array, rho: jax.Array) -> jax.Array:
            sigma = jax.nn.softplus(rho)
            return -0.5 * jnp.sum(1.0 + 2.0 * jnp.log(sigma) - mu**2 - sigma**2)

        return _kl(self.kernel_mu.value, self.kernel_rho.value) + _kl(
            self.bias_mu.value, self.bias_rho.value
        )


BayesConvCls = type[StandardBayesConv1D] | type[FlipoutConv1D]


class BayesCNN1D(nnx.Module):
    def __init__(
        self,
        conv_cls: BayesConvCls,
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
            conv_cls: Bayesian convolution layer class
                (:class:`StandardBayesConv1D` or :class:`FlipoutConv1D`).
            n_filters: Number of convolutional filters. Defaults to 4.
            kernel_size: Size of the convolutional kernel. Defaults to 5.
            act_fn: Activation function. Defaults to ``nnx.gelu``.
            rngs: RNGs for the flax internal modules.
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.act_fn = act_fn

        self.conv = conv_cls(
            in_features=1,
            out_features=n_filters,
            kernel_size=kernel_size,
            padding="VALID",
            rngs=rngs,
        )

        # GaussianBlock to output mean and variance
        self.gaussian_block = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass through the Bayesian CNN.

        Args:
            x: Input with shape ``(batch, 1, window_size)``.
                Will be transposed to ``(batch, window_size, 1)``.
                Accepts variable-length windows.
            rngs: RNGs for weight sampling. The ``params`` stream is used
                to draw a key for the Bayesian convolution layer.

        Returns:
            Concatenated ``[mu, var_positive]`` with shape ``(batch, 2)``.
        """
        # Transpose from (batch, 1, window_size) to (batch, window_size, 1)
        x = jnp.transpose(x, (0, 2, 1))

        # Bayesian Conv1D with activation
        x = self.conv(x, key=rngs.params())
        x = self.act_fn(x)

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=1)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gaussian_block(x)

    def kl_divergence(self) -> jax.Array:
        """Total KL divergence across all Bayesian layers."""
        return self.conv.kl_divergence()


class QAVICNN1D(nnx.Module):
    """1D CNN for QAVI-based RUL prediction with externally provided conv weights.

    Architecture: functional Conv1D (no internal conv parameters) -> Activation ->
    Global Average Pooling -> GaussianBlock.  Convolution kernel and bias are
    generated externally by PQC generators and passed into the forward call.
    Only the :class:`GaussianBlock` head carries trainable parameters.
    """

    def __init__(
        self,
        n_filters: int = 4,
        kernel_size: int = 5,
        act_fn: nnx.Module = nnx.gelu,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise the QAVI CNN.

        Args:
            n_filters: Number of convolutional filters. Defaults to 4.
            kernel_size: Size of the convolutional kernel. Defaults to 5.
            act_fn: Activation function. Defaults to ``nnx.gelu``.
            rngs: RNGs for the GaussianBlock parameters.
        """
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.act_fn = act_fn

        # GaussianBlock to output mean and variance
        self.gaussian_block = GaussianBlock(n_filters, 1, rngs=rngs)

    def __call__(self, x: jax.Array, kernel: jax.Array, bias: jax.Array) -> jax.Array:
        """Forward pass with externally provided conv weights.

        Args:
            x: Input with shape ``(batch, 1, window_size)``.
                Will be transposed to ``(batch, window_size, 1)``.
            kernel: Convolution kernel with shape
                ``(kernel_size, 1, n_filters)``.
            bias: Bias vector with shape ``(n_filters,)``.

        Returns:
            Concatenated ``[mu, var_positive]`` with shape ``(batch, 2)``.
        """
        # Transpose from (batch, 1, window_size) to (batch, window_size, 1)
        x = jnp.transpose(x, (0, 2, 1))

        # Functional Conv1D with externally provided weights
        x = (
            jax.lax.conv_general_dilated(
                x,
                kernel,
                window_strides=(1,),
                padding="VALID",
                dimension_numbers=("NHC", "HIO", "NHC"),
            )
            + bias
        )
        x = self.act_fn(x)

        # Global Average Pooling: (batch, length, n_filters) -> (batch, n_filters)
        x = jnp.mean(x, axis=1)

        # GaussianBlock: (batch, n_filters) -> (batch, 2)
        return self.gaussian_block(x)


class LSTM(nnx.Module):
    """Layers of LSTM and dropout with a final linear layer to output the prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float = 0.1,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.dropout_rate: float = dropout_rate

        # Layers
        self.lstm_layer_1 = nnx.RNN(
            nnx.OptimizedLSTMCell(input_size, hidden_size, rngs=rngs),
            return_carry=True,  # Provide the hidden states for every time step to the next layer.
        )
        self.dropout_1 = nnx.Dropout(dropout_rate)
        self.lstm_layer_2 = nnx.RNN(
            nnx.OptimizedLSTMCell(hidden_size, hidden_size, rngs=rngs),
            return_carry=False,  # Only output the final hidden state.
        )
        self.dropout_2 = nnx.Dropout(dropout_rate)
        self.linear = nnx.Linear(hidden_size, 1, rngs=rngs)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        # x shape: (batch, sequence_length, n_features)
        carry_1, out_1 = self.lstm_layer_1(x)
        x = self.dropout_1(out_1, rngs=rngs)
        out_2 = self.lstm_layer_2(x, initial_carry=carry_1)
        x = self.dropout_2(out_2, rngs=rngs)
        x = self.linear(x)
        return x[:, -1, :]  # Return the output of the last time step (predicted RUL)


def mc_sample(model: RandomCallModel, x: jax.Array, keys: jax.Array) -> jax.Array:
    """Generate MC samples from a model with stochastic forward pass.

    Args:
        model: A model that accepts RNGs in its forward pass (e.g. MC Dropout).
        x: Input data for which to generate predictions, shape (n_x, ...).
        keys: Array of JAX PRNG keys for sampling. Length determines the number
            of MC samples.
    Returns:
        jax.Array: MC samples with shape (n_samples, n_x, n_outputs).
    """

    @nnx.vmap(in_axes=(None, None, 0), out_axes=0)
    def forward(model, x, key):
        return model(x, rngs=nnx.Rngs(key))

    return forward(model, x, keys)


def nll_loss(model: nnx.Module, batch: jax.Array, beta: float = 0.0) -> jax.Array:
    """Negative log-likelihood loss based on a Gaussian predictive distribution.

    Implements the standard NLL (Equation (31) in
    https://doi.org/10.1016/j.ymssp.2023.110796) when ``beta=0``. When
    ``beta>0``, applies variance weighting as proposed in
    https://arxiv.org/pdf/2203.09168 (beta-NLL).

    Args:
        model (nnx.Module): Gaussian neural network with 2 outputs (mean and variance).
        batch (jax.Array): batched input data.
        beta (float): Variance-weighting exponent. ``0.0`` gives standard NLL;
            ``0.5`` is the value recommended in the beta-NLL paper. Defaults to
            ``0.0``.

    Returns:
        jax.Array: loss value for the batch.
    """

    xs, labels = batch
    outputs = model(xs)
    means, variances = outputs[:, 0], outputs[:, 1]
    variances = jnp.clip(variances, min=1e-6)
    losses = 0.5 * jnp.log(variances) + 0.5 * jnp.square(labels - means) / variances

    if beta > 0:
        losses = losses * jax.lax.stop_gradient(variances) ** beta

    return jnp.mean(losses)


def nll_loss_mcd(
    model: nnx.Module,
    batch: jax.Array,
    beta: float = 0.0,
    rngs: Optional[nnx.Rngs] = None,
) -> jax.Array:
    """NLL loss for models that require RNGs at call time (e.g. MC Dropout).

    Same formulation as :func:`nll_loss` but forwards ``rngs`` to the model's
    forward pass so that stochastic layers (dropout) receive fresh random keys.

    Args:
        model (nnx.Module): Gaussian neural network with 2 outputs (mean and variance).
        batch (jax.Array): batched input data.
        beta (float): Variance-weighting exponent. ``0.0`` gives standard NLL;
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

    if beta > 0:
        losses = losses * jax.lax.stop_gradient(variances) ** beta

    return jnp.mean(losses)


def elbo_nll_loss(
    model: nnx.Module,
    batch: jax.Array,
    *,
    rngs: nnx.Rngs,
    n_train: int,
    beta: float = 0.0,
) -> jax.Array:
    """ELBO NLL loss for Bayesian models (Bayes by Backprop).

    Combines the Gaussian NLL data fit with the KL divergence regulariser
    scaled by ``1 / n_train``::

        L = NLL + KL(q ‖ p) / N

    Args:
        model (nnx.Module): Bayesian model with Gaussian output and
            ``kl_divergence()`` method.
        batch (jax.Array): batched input data ``(xs, labels)``.
        rngs (nnx.Rngs): RNGs for weight sampling (forwarded to the model).
        n_train (int): Total number of training samples (for KL scaling).
        beta (float): Variance-weighting exponent. ``0.0`` gives standard NLL;

    Returns:
        jax.Array: scalar ELBO loss value.
    """
    xs, labels = batch
    outputs = model(xs, rngs=rngs)
    means, variances = outputs[:, 0], outputs[:, 1]
    variances = jnp.clip(variances, min=1e-6)

    nll = jnp.mean(
        0.5 * jnp.log(variances) + 0.5 * jnp.square(labels - means) / variances
    )
    if beta > 0:
        nll = nll * jax.lax.stop_gradient(jnp.mean(variances) ** beta)

    kl = model.kl_divergence() / n_train
    return nll + kl


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
