from __future__ import annotations

import logging
import pathlib
import tempfile
import time
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import Callable, Iterable

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import mlflow
import orbax.checkpoint as ocp
import tqdm

from .module import eval_step_simple, train_step_simple

logger = logging.getLogger(__name__)

# The first argument is the batch, the second is the RNG key.
type StepFn = Callable[
    [nnx.Module, tuple[jax.Array, jax.Array], jax.Array, nnx.Optimizer], jax.Array
]

type Callback = Callable[[TrainingPhase, TrainingContext], None]


class TrainingPhase(Enum):
    INIT = auto()
    EPOCH_START = auto()
    EVAL_START = auto()
    EPOCH_END = auto()
    BEFORE_RETURN = auto()


@dataclass
class TrainingContext:
    epoch: int
    train_loss: float
    val_loss: float
    best_val_loss: float
    model: nnx.Module
    optimizer: nnx.Optimizer
    model_best_state: nnx.State


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


# ==================================================
# ============= List of callbacks ==================
# ==================================================


class PrintReporter:
    def __init__(self, print_every: int = 1) -> None:
        self.print_every = print_every

    def __call__(self, phase: TrainingPhase, context: TrainingContext) -> None:
        if phase == TrainingPhase.EPOCH_END and context.epoch % self.print_every == 0:
            print(
                f"Epoch {context.epoch:3d} | "
                f"Train Loss: {context.train_loss:.6f} | "
                f"Val Loss: {context.val_loss:.6f} | "
                f"Best Val Loss: {context.best_val_loss:.6f}"
            )


class LogReporter:
    def __init__(self, log_every: int = 1) -> None:
        self.log_every = log_every

    def __call__(self, phase: TrainingPhase, context: TrainingContext) -> None:
        if phase == TrainingPhase.EPOCH_END and context.epoch % self.log_every == 0:
            logger.info(
                f"Epoch {context.epoch:3d} | "
                f"Train Loss: {context.train_loss:.6f} | "
                f"Val Loss: {context.val_loss:.6f} | "
                f"Best Val Loss: {context.best_val_loss:.6f}"
            )


def mlflow_track_model_best_state(
    phase: TrainingPhase, context: TrainingContext
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


def mlflow_track_losses(phase: TrainingPhase, context: TrainingContext) -> None:
    if phase == TrainingPhase.EPOCH_END:
        mlflow.log_metric("train_loss", context.train_loss, step=context.epoch)
        mlflow.log_metric("val_loss", context.val_loss, step=context.epoch)
        mlflow.log_metric("best_val_loss", context.best_val_loss, step=context.epoch)


# ==================================================
# ================= Train Loop =====================
# ==================================================


def train_loop(
    n_epochs: int,
    dataloader_train: Iterable,
    dataloader_val: Iterable,
    initial_key: jax.Array,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    train_batch_fn: StepFn = train_step_simple,
    eval_batch_fn: StepFn = eval_step_simple,
    callbacks: Iterable[Callback] = (LogReporter(),),
    early_stopper: EarlyStopper | None = None,
) -> None:
    def run_callbacks(phase: TrainingPhase, info: TrainingContext) -> None:
        for callback in callbacks:
            callback(phase, info)

    phase = TrainingPhase.INIT
    context = TrainingContext(
        epoch=0,
        train_loss=float("inf"),
        val_loss=float("inf"),
        best_val_loss=float("inf"),
        model=model,
        optimizer=optimizer,
        model_best_state=jax.tree.map(lambda x: x, nnx.state(model, nnx.Param)),
    )
    run_callbacks(phase, context)

    key = initial_key

    epoch = 0
    try:
        for epoch in tqdm.tqdm(range(n_epochs), desc="training"):
            phase = TrainingPhase.EPOCH_START
            context.epoch = epoch
            run_callbacks(phase, context)

            model.train()
            train_losses = []
            for batch in dataloader_train:
                splits = jax.random.split(key, num=batch[0].shape[0] + 1)
                key, subkeys = splits[0], splits[1:]
                loss = train_batch_fn(model, batch, subkeys, optimizer)
                train_losses.append(loss)

            phase = TrainingPhase.EVAL_START
            context.train_loss = jnp.mean(jnp.array(train_losses)).item()
            run_callbacks(phase, context)

            model.eval()
            val_losses = []
            for batch in dataloader_val:
                splits = jax.random.split(key, num=batch[0].shape[0] + 1)
                key, subkeys = splits[0], splits[1:]
                val_losses.append(eval_batch_fn(model, batch, subkeys, optimizer))

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
