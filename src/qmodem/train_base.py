from __future__ import annotations

import dataclasses
import pathlib
import tempfile
import time
from enum import Enum, StrEnum, auto
from typing import Callable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import mlflow
import orbax.checkpoint as ocp


class TrainingPhase(Enum):
    INIT = auto()
    EPOCH_START = auto()
    EVAL_START = auto()
    EPOCH_END = auto()
    BEFORE_RETURN = auto()


@dataclasses.dataclass
class BaseTrainingContext:
    epoch: int
    val_loss: float
    best_val_loss: float
    model: nnx.Module  # Replace with the actual model type if known
    model_best_state: nnx.State  # Replace with the actual model state type if known


type Callback = Callable[[TrainingPhase, BaseTrainingContext], None]


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


# Callbacks


def mlflow_track_model_best_state(
    phase: TrainingPhase, context: BaseTrainingContext
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


class OutputVarianceTracker:
    def __init__(
        self, base_key: jax.Array, X_batch: jax.Array, n_samples: int = 100
    ) -> None:
        """Callback to track the variance of the predictive mean over a batch of
        inputs."""
        self.key = base_key
        self.X_batch = X_batch
        self.n_samples = n_samples

    def _sample(self, model: nnx.Module, key: jax.Array) -> jax.Array:
        # Sample predictions from the model using the provided key
        return model(self.X_batch, rngs=nnx.Rngs(default=key))

    def __call__(self, phase: TrainingPhase, context: BaseTrainingContext) -> None:
        if phase == TrainingPhase.EPOCH_END:
            context.model.train()  # e.g. MCD

            preds = []
            for _ in range(self.n_samples):
                self.key, subkey = jax.random.split(self.key, num=2)
                preds.append(self._sample(context.model, subkey))

            preds = jnp.stack(preds, axis=0)  # Shape (n_samples, batch_size)
            var_preds = jnp.var(preds, axis=0)  # Variance across samples for each input
            mean_var_preds = jnp.mean(
                var_preds, axis=0
            )  # Mean variance across the batch

            context.model.eval()
            mlflow.log_metric(
                "predictive_mean_variance",
                mean_var_preds[0],
                step=context.epoch,
            )
            mlflow.log_metric(
                "predictive_std_variance",
                mean_var_preds[1],
                step=context.epoch,
            )
