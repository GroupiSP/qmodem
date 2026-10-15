from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
from flax import nnx

from qmodem.module import nll_batched
from qmodem.train import StepFn


@dataclasses.dataclass(frozen=True)
class StandardStepFactoryContext:
    n_train_samples: int


type StandardStepFactory = Callable[[StandardStepFactoryContext], tuple[StepFn, StepFn]]


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
