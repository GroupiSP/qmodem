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


class ConvWeightGenerator(Protocol):
    def init_params(self, rngs: nnx.Rngs) -> None: ...
    def __call__(self, rngs: nnx.Rngs) -> jax.Array: ...


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
