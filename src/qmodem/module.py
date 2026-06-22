from __future__ import annotations

import logging
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx

logger = logging.getLogger(__name__)


class RandomCallModel(Protocol):
    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array: ...


class PQC(Protocol):
    n_qubits: int
    params_shape: tuple[int, ...]

    def __call__(self, x: jax.Array, params: jax.Array) -> jax.Array: ...


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


class PQCModule(nnx.Module):
    def __init__(self, quantum_circuit: PQC, rngs: nnx.Rngs):
        self.quantum_circuit = quantum_circuit
        # Variational parameters of the PQC (shape depends on the circuit design)
        self.params = nnx.Param(
            jax.random.uniform(
                rngs.params(),
                shape=quantum_circuit.params_shape,
                minval=0.0,
                maxval=2 * jnp.pi,
            )
        )

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        return jnp.array(self.quantum_circuit(x, self.params.value))


class PQCLinearPPGenerator(nnx.Module):
    def __init__(self, quantum_circuit: PQC, n_out_linear: int, rngs: nnx.Rngs):
        self.pqc_module = PQCModule(quantum_circuit, rngs)
        self.post_processor = nnx.Linear(
            quantum_circuit.n_qubits, n_out_linear, rngs=rngs
        )

    def __call__(self, rngs: nnx.Rngs) -> jax.Array:
        # Sample x from a uniform [0, 2\pi] distribution
        x = jax.random.uniform(
            rngs.params(),
            shape=(),  # TODO: can the shape be more general?
            minval=0,
            maxval=2 * jnp.pi,
        )
        pqc_out = self.pqc_module(x, rngs)
        return self.post_processor(pqc_out)


class PQCConv1D(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        padding: str,
        quantum_circuit: PQC,
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
        self.generators = nnx.List(
            [
                PQCLinearPPGenerator(
                    quantum_circuit, n_out_linear=quantum_circuit.n_qubits, rngs=rngs
                )  # linear pp does not change the output dimension of the PQC.
                for _ in range(out_features + 1)
            ]
        )  # +1 for bias

        self._kernel_shape = (kernel_size, in_features, out_features)
        self._bias_shape = (out_features,)

    def __call__(self, x: jax.Array, rngs: nnx.Rngs) -> jax.Array:
        """Forward pass: generate one set of weights and convolve the batch.

        Args:
            x: Input with shape (batch, length, in_features).
            rngs: RNGs for weight generation.
        """
        # Generate kernels and bias from the PQC generators
        key = rngs.params()
        keys = jax.random.split(key, len(self.generators))
        kernels_and_bias = [
            gen(nnx.Rngs(params=keys[i])) for i, gen in enumerate(self.generators)
        ]
        kernel = jnp.stack(kernels_and_bias[:-1], axis=-1).reshape(self._kernel_shape)
        bias = kernels_and_bias[-1][: self.out_features]  # EV of the first 4 qubits

        out = jax.lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1,),
            padding=self.padding,
            dimension_numbers=("NHC", "HIO", "NHC"),
        )
        return out + bias


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


def negative_log_likelihood(
    model: nnx.Module,
    batch: tuple[jax.Array, jax.Array],
    rngs: nnx.Rngs,
    beta: float = 0.0,
) -> jax.Array:
    """Gaussian NLL loss for heteroscedastic regression.

    Args:
       model (nnx.Module): Heteroscedastic regression model with Gaussian output.
       batch (tuple[jax.Array, jax.Array]): Batched input data (xs, labels).
       rngs (nnx.Rngs): RNGs for stochastic forward pass (if applicable).
       beta (float): Variance-weighting exponent. Implements the beta-NLL loss in arXiv:2203.09168.
           ``0.0`` gives standard NLL.

    Returns:
        jax.Array: Per-sample NLL losses with shape (batch,).
    """

    xs, labels = batch
    # Add a batch dimension to xs for the model's forward pass
    xs_b = jnp.expand_dims(xs, axis=0)
    outputs = model(xs_b, rngs=rngs)
    means, variances = outputs[:, 0], outputs[:, 1]
    variances = jnp.clip(variances, min=1e-6)
    losses = 0.5 * jnp.log(variances) + 0.5 * jnp.square(labels - means) / variances

    if beta > 0:
        # TODO: This implementation of the beta-NLL loss might be wrong. Revisit.
        logger.warning(
            "The beta-NLL loss implementation is untested and might be incorrect for beta > 0. Please use beta=0."
        )
        losses = losses * jax.lax.stop_gradient(variances) ** beta

    return losses


def _per_sample_squared_error(
    model: nnx.Module,
    sample: tuple[jax.Array, jax.Array],
    sample_key: jax.Array,
) -> jax.Array:
    xs, labels = sample
    # Add a batch dimension to xs for the model's forward pass
    xs_b = jnp.expand_dims(xs, axis=0)
    outputs = model(xs_b, rngs=nnx.Rngs(sample_key))
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
        err_fn = nnx.vmap(_per_sample_squared_error, in_axes=(None, 0, 0), out_axes=0)
        return jnp.mean(err_fn(model, batch, keys))

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
        err_fn = nnx.vmap(_per_sample_squared_error, in_axes=(None, 0, 0), out_axes=0)
        return jnp.mean(err_fn(model, batch, keys))

    return loss_fn(model)
