from __future__ import annotations

import logging
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx

logger = logging.getLogger(__name__)


class PQC(Protocol):
    n_qubits: int
    params_shape: tuple[int, ...]

    def __call__(self, x: jax.Array, params: jax.Array) -> jax.Array: ...


@nnx.vmap(in_axes=(None, 0, 0), out_axes=0)
def model_fwd(model: nnx.Module, x_i: jax.Array, key: jax.Array) -> jax.Array:
    # NOTE: we need to add a batch dimension to x_i since the model expects a batch of inputs.
    # NOTE: we need to remove the batch dimension from the output since we only want the output for the single input x_i.
    return model(x_i[None], rngs=nnx.Rngs(default=key))[0]


@nnx.vmap(in_axes=(None, None, 0, 0), out_axes=0)
def mc_sample(
    model: nnx.Module, x: jax.Array, key_weights: jax.Array, key_noise: jax.Array
) -> jax.Array:
    """Apply the model to a single input x with as many keys as the number of Monte
    Carlo samples."""
    mu, var = model(x, rngs=nnx.Rngs(default=key_weights))[0]  # Shape (2,)
    return mu + jnp.sqrt(var) * jax.random.normal(key_noise, shape=(1,))


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

        # Variational parameters
        self.kernel_mu = nnx.Param(jax.random.normal(rngs.params(), k_shape) * 0.1)
        self.kernel_rho = nnx.Param(jnp.full(k_shape, -3.0))
        self.bias_mu = nnx.Param(jnp.zeros(out_features))
        self.bias_rho = nnx.Param(jnp.full(out_features, -3.0))

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass: sample one set of weights and convolve the batch.

        Args:
            x: Input with shape ``(batch, length, in_features)``.
            rngs: Flax RNGs for weight sampling.

        Returns:
            Convolved output with shape ``(batch, L_out, out_features)``.
        """
        k1, k2 = jax.random.split(rngs.params(), 2)
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

    def mean_posterior_variance(self) -> jax.Array:
        """Mean of the posterior variance across all weights."""
        k_var = jnp.square(jax.nn.softplus(self.kernel_rho.value))
        b_var = jnp.square(jax.nn.softplus(self.bias_rho.value))
        return jnp.mean(jnp.concatenate([k_var.flatten(), b_var.flatten()]))


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

        # Variational parameters
        self.kernel_mu = nnx.Param(jax.random.normal(rngs.params(), k_shape) * 0.1)
        self.kernel_rho = nnx.Param(jnp.full(k_shape, -3.0))
        self.bias_mu = nnx.Param(jnp.zeros(out_features))
        self.bias_rho = nnx.Param(jnp.full(out_features, -3.0))

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass with per-sample sign-flipped perturbations.

        Args:
            x: Input with shape ``(batch, length, in_features)``.
            rngs: RNGs for weight and sign sampling.

        Returns:
            Convolved output with shape ``(batch, L_out, out_features)``.
        """
        k1, k2, k3, k4 = jax.random.split(rngs.params(), 4)
        batch_size = x.shape[0]
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
        s = jax.random.rademacher(k3, (batch_size, 1, self.in_features)).astype(x.dtype)
        r = jax.random.rademacher(k4, (batch_size, 1, self.out_features)).astype(
            x.dtype
        )

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
        """KL(q ‖ p) with unit-normal prior p = N(0, 1).

        Has an analytical form
        """

        def _kl(mu: jax.Array, rho: jax.Array) -> jax.Array:
            sigma = jax.nn.softplus(rho)
            return -0.5 * jnp.sum(1.0 + 2.0 * jnp.log(sigma) - mu**2 - sigma**2)

        return _kl(self.kernel_mu.value, self.kernel_rho.value) + _kl(
            self.bias_mu.value, self.bias_rho.value
        )

    def mean_posterior_variance(self) -> jax.Array:
        """Mean of the posterior variance across all weights."""
        k_var = jnp.square(jax.nn.softplus(self.kernel_rho.value))
        b_var = jnp.square(jax.nn.softplus(self.bias_rho.value))
        return jnp.mean(jnp.concatenate([k_var.flatten(), b_var.flatten()]))


class ConvWeightGenerator(Protocol):
    def init_params(self, rngs: nnx.Rngs) -> None: ...
    def __call__(self, rngs: nnx.Rngs) -> jax.Array: ...


class GeneratorConv1D(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        padding: str,
        generator: ConvWeightGenerator,
        *,
        rngs: nnx.Rngs,
    ):
        """Bayesian 1D convolutional layer, in which the kernels and biases are
        generated by a PQC.

        Specifically, the weight generator is a sequence of a PQC and a linear layer
        which maintains the output dimension of the PQC. One generator is used for each
        filter of the convolutional layer and one more generator.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding
        self.generator = generator

        self.generator.init_params(rngs)

        # Convolutional bias (deterministic)
        self.b_conv = nnx.Param(jnp.zeros((self.out_features,)))

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass: generate one set of weights and convolve the batch.

        Args:
            x: Input with shape (batch, length, in_features).
            rngs: RNGs for weight generation.
        """
        w_conv = self.generator(rngs)

        out = jax.lax.conv_general_dilated(
            x,
            w_conv,
            window_strides=(1,),
            padding=self.padding,
            dimension_numbers=("NHC", "HIO", "NHC"),
        )
        return out + self.b_conv


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


def nll_batched(
    model: nnx.Module,
    batch: tuple[jax.Array, jax.Array],
    keys: jax.Array,
    beta: float = 0.0,
) -> jax.Array:
    """Gaussian NLL loss for heteroscedastic regression.

    Args:
       model (nnx.Module): Regression model with Gaussian output.
       batch (tuple[jax.Array, jax.Array]): Batched input data (xs, labels).
       keys (jax.Array): Array of JAX PRNG keys for sampling.
       beta (float): Variance-weighting exponent. Implements the beta-NLL loss in arXiv:2203.09168.
           ``0.0`` gives standard NLL.

    Returns:
        jax.Array: NLL losses with shape (batch,).
    """

    xs, labels = batch[0], batch[1].squeeze(-1)
    # Add a batch dimension to xs for the model's forward pass
    outputs = model_fwd(model, xs, keys)
    means, variances = outputs[:, 0], outputs[:, 1]
    variances = jnp.clip(variances, min=1e-8)
    losses = 0.5 * jnp.log(variances) + 0.5 * jnp.square(labels - means) / variances

    # beta-NLL loss (https://arxiv.org/abs/2203.09168)
    w = jax.lax.stop_gradient(variances) ** beta
    losses = (
        losses * w / jnp.mean(w)
    )  # Normalize by the mean of the weights to keep the loss scale consistent across different beta values.

    return losses


def squared_error_batched(
    model: nnx.Module,
    batch: tuple[jax.Array, jax.Array],
    keys: jax.Array,
) -> jax.Array:
    xs, labels = batch[0], batch[1].squeeze(-1)
    outputs = model_fwd(model, xs, keys)
    # Assumes that when the output layer has two elements, the first one is the mean prediction
    losses = jnp.square(outputs[:, 0] - labels)

    return losses


def train_step_simple(
    model: nnx.Module,
    batch: tuple[jax.Array, jax.Array],
    keys: jax.Array,
    optimizer: nnx.Optimizer,
) -> jax.Array:
    """Single training step with MSE loss."""

    def loss_fn(model):
        return jnp.mean(squared_error_batched(model, batch, keys))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def eval_step_simple(
    model: nnx.Module,
    batch: tuple[jax.Array, jax.Array],
    keys: jax.Array,
    optimizer: nnx.Optimizer,
) -> jax.Array:
    """Single evaluation step with MSE loss."""

    def loss_fn(model):
        return jnp.mean(squared_error_batched(model, batch, keys))

    return loss_fn(model)
