from __future__ import annotations

import logging
import pathlib
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import mlflow
import orbax.checkpoint as ocp
import tqdm

from .module import eval_step_simple
from .train_base import BaseTrainingContext, Callback, EarlyStopper, TrainingPhase

logger = logging.getLogger(__name__)


class TrainStepFn(Protocol):
    def __call__(
        self,
        model: nnx.Module,
        discriminator: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,  # can be either the generator or discriminator optimizer, depending on the step function
    ) -> jax.Array: ...


class EvalStepFn(Protocol):
    def __call__(
        self,
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
        keys: jax.Array,
        optimizer: nnx.Optimizer,  # generally unused at eval time. Included for symmetry.
    ) -> jax.Array: ...


@dataclass
class AdversarialTrainingContext(BaseTrainingContext):
    generator_loss: float
    discriminator_loss: float
    discriminator: nnx.Module
    optimizer_generator: nnx.Optimizer
    optimizer_discriminator: nnx.Optimizer


# ==================================================
# ============= List of callbacks ==================
# ==================================================


class PrintReporter:
    def __init__(self, print_every: int = 1) -> None:
        self.print_every = print_every

    def __call__(
        self, phase: TrainingPhase, context: AdversarialTrainingContext
    ) -> None:
        if phase == TrainingPhase.EPOCH_END and context.epoch % self.print_every == 0:
            print(
                f"Epoch {context.epoch:3d} | "
                f"Generator Loss: {context.generator_loss:.6f} | "
                f"Discriminator Loss: {context.discriminator_loss:.6f} | "
                f"Val Loss: {context.val_loss:.6f} | "
                f"Best Val Loss: {context.best_val_loss:.6f}"
            )


class LogReporter:
    def __init__(self, log_every: int = 1) -> None:
        self.log_every = log_every

    def __call__(
        self, phase: TrainingPhase, context: AdversarialTrainingContext
    ) -> None:
        if phase == TrainingPhase.EPOCH_END and context.epoch % self.log_every == 0:
            logger.info(
                f"Epoch {context.epoch:3d} | "
                f"Generator Loss: {context.generator_loss:.6f} | "
                f"Discriminator Loss: {context.discriminator_loss:.6f} | "
                f"Val Loss: {context.val_loss:.6f} | "
                f"Best Val Loss: {context.best_val_loss:.6f}"
            )


def mlflow_track_model_best_state(
    phase: TrainingPhase, context: AdversarialTrainingContext
) -> None:
    if phase == TrainingPhase.BEFORE_RETURN:
        run = mlflow.active_run()
        if run is None:
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = pathlib.Path(tmp_dir) / "best_model_state"
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(ckpt_path, context.model_best_state)
            time.sleep(0.1)  # let Orbax finish async writes
            mlflow.log_artifacts(str(ckpt_path), artifact_path="best_model_state")


def mlflow_track_losses(
    phase: TrainingPhase, context: AdversarialTrainingContext
) -> None:
    if phase == TrainingPhase.EPOCH_END:
        mlflow.log_metric("generator_loss", context.generator_loss, step=context.epoch)
        mlflow.log_metric(
            "discriminator_loss", context.discriminator_loss, step=context.epoch
        )
        mlflow.log_metric("val_loss", context.val_loss, step=context.epoch)
        mlflow.log_metric("best_val_loss", context.best_val_loss, step=context.epoch)


# ==================================================
# ================= Train Loop =====================
# ==================================================


def train_loop(
    n_epochs: int,
    train_dataloader_builder: Callable[[int], Iterable],
    val_dataloader_builder: Callable[[int], Iterable],
    initial_key: jax.Array,
    model: nnx.Module,
    discriminator: nnx.Module,
    optimizer_generator: nnx.Optimizer,
    optimizer_discriminator: nnx.Optimizer,
    generator_batch_fn: TrainStepFn,  # TODO: add default
    discriminator_batch_fn: TrainStepFn,  # TODO: add default
    eval_batch_fn: EvalStepFn = eval_step_simple,
    callbacks: Iterable[Callback] = (LogReporter(),),
    early_stopper: EarlyStopper | None = None,
) -> None:
    """Train loop for adversarial training of a generative supervised model and a
    discriminator.

    Args:
        n_epochs: Maximum number of epochs to train for.
        dataloader_train: Iterable of training batches. Each batch is a tuple of (inputs, targets).
        dataloader_val: Iterable of validation batches. Each batch is a tuple of (inputs, targets).
        initial_key: Initial JAX PRNG key for random number generation.
        model: The generative supervised model to be trained.
        discriminator: The discriminator model to be trained adversarially against the generative model.
        optimizer_generator: Optimizer for the generative model.
        optimizer_discriminator: Optimizer for the discriminator model.
        generator_batch_fn: Function that performs a training step for the generator. Should return the generator loss for the batch.
        discriminator_batch_fn: Function that performs a training step for the discriminator. Should return the discriminator loss for the batch.
        eval_batch_fn: Function that evaluates the generative model on a validation batch. Should return the validation loss for the batch.
        callbacks: Iterable of callback functions to be called at different phases of training.
        early_stopper: Optional EarlyStopper instance to enable early stopping based on validation loss.
    """

    def run_callbacks(phase: TrainingPhase, info: AdversarialTrainingContext) -> None:
        for callback in callbacks:
            callback(phase, info)

    phase = TrainingPhase.INIT
    context = AdversarialTrainingContext(
        epoch=0,
        generator_loss=float("inf"),
        discriminator_loss=float(
            "inf"
        ),  # discrimination is traditionally a maximization problem, but here we work with losses.
        val_loss=float("inf"),
        best_val_loss=float("inf"),
        model=model,  # generative supervised model
        discriminator=discriminator,
        optimizer_generator=optimizer_generator,
        optimizer_discriminator=optimizer_discriminator,
        model_best_state=jax.tree.map(lambda x: x, nnx.state(model, nnx.Param)),
    )
    run_callbacks(phase, context)

    key = initial_key

    epoch = 0
    try:
        discriminator.train()  # discriminator is always in training mode.

        for epoch in tqdm.tqdm(range(n_epochs), desc="training"):
            phase = TrainingPhase.EPOCH_START
            context.epoch = epoch
            run_callbacks(phase, context)

            model.train()
            generator_losses = []
            discriminator_losses = []
            dataloader_train = train_dataloader_builder(epoch)
            for batch in dataloader_train:
                splits = jax.random.split(key, num=batch[0].shape[0] + 1)
                key, subkeys = splits[0], splits[1:]
                generator_loss = generator_batch_fn(
                    model, discriminator, batch, subkeys, optimizer_generator
                )

                splits = jax.random.split(key, num=batch[0].shape[0] + 1)
                key, subkeys = splits[0], splits[1:]
                discriminator_loss = discriminator_batch_fn(
                    model, discriminator, batch, subkeys, optimizer_discriminator
                )

                generator_losses.append(generator_loss)
                discriminator_losses.append(discriminator_loss)

            phase = TrainingPhase.EVAL_START
            context.generator_loss = jnp.mean(jnp.array(generator_losses)).item()
            context.discriminator_loss = jnp.mean(
                jnp.array(discriminator_losses)
            ).item()

            run_callbacks(phase, context)

            model.eval()
            val_losses = []
            dataloader_val = val_dataloader_builder(epoch)
            for batch in dataloader_val:
                splits = jax.random.split(key, num=batch[0].shape[0] + 1)
                key, subkeys = splits[0], splits[1:]
                val_losses.append(
                    eval_batch_fn(model, batch, subkeys, optimizer_generator)
                )

            val_loss = jnp.mean(jnp.array(val_losses)).item()

            phase = TrainingPhase.EPOCH_END
            context.val_loss = val_loss
            if val_loss < context.best_val_loss:
                context.best_val_loss = val_loss
                context.model_best_state = jax.tree.map(
                    lambda x: x, nnx.state(model, nnx.Param)
                )
            run_callbacks(phase, context)

            if early_stopper is not None and early_stopper(val_loss):
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}")

    # TODO: replace with callback
    print("=" * 70)
    print(f"Training complete! Best validation loss: {context.best_val_loss:.6f}")
    print()

    phase = TrainingPhase.BEFORE_RETURN
    run_callbacks(phase, context)

    return
