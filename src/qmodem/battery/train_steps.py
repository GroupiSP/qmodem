from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx

from qmodem.module import model_fwd, nll_batched
from qmodem.train import StepFn
from qmodem.train_adversarial import EvalStepFn, TrainStepFn


@dataclasses.dataclass(frozen=True)
class StandardStepFactoryContext:
    n_train_samples: int


type StandardStepFactory = Callable[[StandardStepFactoryContext], tuple[StepFn, StepFn]]


class BayesianModel(Protocol):
    def kl_divergence(self) -> jax.Array: ...


def make_nll_steps(beta: float = 0.0) -> tuple[StepFn, StepFn]:
    @nnx.jit
    def train_step(
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            return jnp.mean(nll_batched(model, batch, keys, beta=beta))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        return jnp.mean(nll_batched(model, batch, keys, beta=beta))

    return train_step, eval_step


def make_elbo_steps(
    context: StandardStepFactoryContext,
) -> tuple[StepFn, StepFn]:
    def elbo_loss(
        model: BayesianModel,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
    ) -> jax.Array:
        return jnp.mean(nll_batched(model, batch, keys)) + (
            model.kl_divergence() / context.n_train_samples
        )

    @nnx.jit
    def train_step(
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        loss, grads = nnx.value_and_grad(elbo_loss)(model, batch, keys)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def eval_step(
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        return elbo_loss(model, batch, keys)

    return train_step, eval_step


def make_qavi_steps(
    beta: float = 0.0,
    adversarial_loss_weight: float = 0.1,
) -> tuple[TrainStepFn, TrainStepFn, EvalStepFn]:
    @nnx.jit
    def generator_step(
        model: nnx.Module,
        discriminator: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(model: nnx.Module) -> jax.Array:
            epsilon = 1e-8
            xs = batch[0]
            batch_size = len(xs)
            model_key, sample_key, discriminator_key, nll_key = jax.random.split(
                key, num=4
            )

            model_out = model_fwd(
                model, xs, jax.random.split(model_key, num=batch_size)
            )
            means, variances = model_out[:, 0], model_out[:, 1]
            standard_deviations = jnp.sqrt(jnp.clip(variances, min=epsilon))
            predictions = (
                jax.random.normal(sample_key, shape=(batch_size,)) * standard_deviations
                + means
            )

            xs_flat = xs.reshape((batch_size, -1))
            probabilities_fake = discriminator(
                jnp.concatenate([xs_flat, predictions[:, None]], axis=1),
                nnx.Rngs(params=discriminator_key),
            )
            adversarial_error = -jnp.log(
                jnp.clip(probabilities_fake, epsilon, 1 - epsilon)
            ).squeeze(-1)
            nll = nll_batched(
                model,
                batch,
                jax.random.split(nll_key, num=batch_size),
                beta=beta,
            )
            return jnp.mean(adversarial_loss_weight * adversarial_error + nll)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    @nnx.jit
    def discriminator_step(
        model: nnx.Module,
        discriminator: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        key: jax.Array,
        optimizer: nnx.Optimizer,
    ) -> jax.Array:
        def loss_fn(discriminator: nnx.Module, model: nnx.Module) -> jax.Array:
            epsilon = 1e-8
            xs, targets = batch[0], batch[1].squeeze(-1)
            batch_size = len(xs)
            model_key, sample_key, discriminator_key = jax.random.split(key, num=3)

            model_out = model_fwd(
                model, xs, jax.random.split(model_key, num=batch_size)
            )
            means, variances = model_out[:, 0], model_out[:, 1]
            standard_deviations = jnp.sqrt(jnp.clip(variances, min=epsilon))
            predictions = (
                jax.random.normal(sample_key, shape=(batch_size,)) * standard_deviations
                + means
            )

            xs_flat = xs.reshape((batch_size, -1))
            rngs = nnx.Rngs(params=discriminator_key)
            probabilities_real = discriminator(
                jnp.concatenate([xs_flat, targets[:, None]], axis=1), rngs
            )
            probabilities_fake = discriminator(
                jnp.concatenate([xs_flat, predictions[:, None]], axis=1), rngs
            )
            error = -jnp.log(probabilities_real + epsilon) - jnp.log(
                1 - probabilities_fake + epsilon
            )
            return jnp.mean(error.squeeze(-1))

        loss, grads = nnx.value_and_grad(loss_fn)(discriminator, model)
        optimizer.update(discriminator, grads)
        return loss

    _, eval_step = make_nll_steps(beta=beta)
    return generator_step, discriminator_step, eval_step
