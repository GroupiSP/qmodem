from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import StrEnum, auto
from typing import Any

import jax
import jax.numpy as jnp


class EarlyStopState(StrEnum):
    WAITING_FOR_IMPROVEMENT = auto()
    IMPROVEMENT_FOUND = auto()
    STOPPED = auto()


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.current_epoch = 0

        self._state = EarlyStopState.WAITING_FOR_IMPROVEMENT

    @property
    def state(self) -> EarlyStopState:
        return self._state

    def __call__(self, current_loss: jax.Array) -> bool:
        self.current_epoch += 1
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            self._state = EarlyStopState.IMPROVEMENT_FOUND
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self._state = EarlyStopState.STOPPED
                print(
                    f"Early stopping triggered at epoch {self.current_epoch}. Validation loss: {current_loss:.4f}"
                )
                return True
            return False


def train_loop(
    n_epochs: int,
    dataloader_train: Iterable,
    dataloader_val: Iterable,
    train_batch_fn: Callable[[Any], jax.Array],
    eval_batch_fn: Callable[[Any], jax.Array],
    *,
    early_stopper: EarlyStopper | None = None,
    print_every: int = 1,
    on_train_epoch_start: Callable[[], None] | None = None,
    on_val_epoch_start: Callable[[], None] | None = None,
    on_validation_improvement: Callable[[], None] | None = None,
) -> tuple[float, int]:
    """Run a training loop with optional early stopping and graceful interruption.

    Iterates for up to ``n_epochs`` epochs.  Each epoch:

    1. Calls ``on_train_epoch_start`` (if provided).
    2. Calls ``train_batch_fn`` for every batch in ``dataloader_train``.
    3. Evaluates train loss by calling ``eval_batch_fn`` on ``dataloader_train``.
    4. Calls ``on_val_epoch_start`` (if provided).
    5. Evaluates validation loss by calling ``eval_batch_fn`` on ``dataloader_val``.
    6. Prints progress every ``print_every`` epochs and on the first epoch.
    7. Checks early stopping if an ``early_stopper`` is provided.

    A ``KeyboardInterrupt`` exits the loop gracefully: the training status at
    the last completed epoch is reported and execution resumes after the call.

    Args:
        n_epochs: Maximum number of training epochs.
        dataloader_train: Training data loader (re-iterable for loss evaluation).
        dataloader_val: Validation data loader (iterable).
        train_batch_fn: Callable that takes a batch and performs one training step.
        eval_batch_fn: Callable that takes a batch and returns a scalar loss value.
        early_stopper: Optional :class:`EarlyStopper` instance.
        print_every: Print progress every this many epochs (also prints epoch 1).
        on_train_epoch_start: Optional callback called before the training phase.
        on_val_epoch_start: Optional callback called before the validation phase.
        on_validation_improvement: Optional callback called when validation loss improves.
    Returns:
        A tuple ``(best_val_loss, epochs_completed)`` where ``best_val_loss`` is
        the lowest validation loss observed and ``epochs_completed`` is the number
        of epochs that finished before stopping.
    """
    best_val_loss = float("inf")
    epochs_completed = 0

    try:
        for epoch in range(n_epochs):
            if on_train_epoch_start is not None:
                on_train_epoch_start()

            train_losses = []
            for batch in dataloader_train:
                loss = train_batch_fn(batch)
                train_losses.append(loss)

            if on_val_epoch_start is not None:
                on_val_epoch_start()

            val_losses = []
            for batch in dataloader_val:
                val_losses.append(eval_batch_fn(batch))

            train_loss = jnp.mean(jnp.array(train_losses))
            val_loss = jnp.mean(jnp.array(val_losses))

            epochs_completed = epoch + 1

            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{n_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                if on_validation_improvement is not None:
                    on_validation_improvement()

            if early_stopper is not None and early_stopper(val_loss):
                break

    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epochs_completed}")

    print("=" * 70)
    print(f"Training complete! Best validation loss: {best_val_loss:.6f}")
    print()

    return best_val_loss, epochs_completed
