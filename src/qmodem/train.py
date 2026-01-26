from __future__ import annotations

from typing import Sequence

import jax


def early_stop(
    val_losses: Sequence[jax.Array],
    patience: int,
    min_delta: float = 0.0,
) -> bool:
    """Check if early stopping criterion is met.

    Args:
        val_losses (Sequence[float]): List of validation losses.
        patience (int): Number of epochs to wait for improvement.
        min_delta (float, optional): Minimum change to qualify as an improvement. Defaults to 0.0.

    Returns:
        bool: True if early stopping criterion is met, False otherwise.
    """
    current_epoch = len(val_losses) + 1
    if current_epoch < patience:
        return False

    recent_losses = val_losses[-(patience + 1) :]
    best_loss = min(recent_losses[:-1])
    current_loss = recent_losses[-1]

    if current_loss > best_loss - min_delta:
        print(f"Early stopping triggered at epoch {current_epoch}.")
        return True
    return False
