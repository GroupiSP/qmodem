from __future__ import annotations

from dataclasses import dataclass, field

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax
import pytest

from qmodem.train import PrintReporter, TrainingContext, train_loop
from qmodem.train_base import EarlyStopper, TrainingPhase

type Batch = tuple[jax.Array, jax.Array]


def _make_dataloader(
    n_batches: int, value: float = 1.0, batch_size: int = 2
) -> list[Batch]:
    batch = (
        jnp.full((batch_size, 1), value),
        jnp.full((batch_size, 1), value),
    )
    return [batch] * n_batches


def _identity_step(
    model: nnx.Module,
    batch: Batch,
    keys: jax.Array,
    optimizer: nnx.Optimizer,
) -> jax.Array:
    return jnp.mean(batch[1])


@dataclass
class ContextRecorder:
    phases: list[TrainingPhase] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    best_val_losses: list[float] = field(default_factory=list)

    def __call__(self, phase: TrainingPhase, context: TrainingContext) -> None:
        self.phases.append(phase)
        if phase == TrainingPhase.EPOCH_END:
            self.epochs.append(context.epoch)
        if phase == TrainingPhase.BEFORE_RETURN:
            self.best_val_losses.append(context.best_val_loss)


def _run_train_loop(
    *,
    n_epochs: int,
    dataloader: list[Batch],
    train_batch_fn=_identity_step,
    eval_batch_fn=_identity_step,
    callbacks=(),
    early_stopper: EarlyStopper | None = None,
) -> None:
    model = nnx.Linear(1, 1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)
    train_loop(
        n_epochs=n_epochs,
        train_dataloader_builder=lambda epoch: dataloader,
        val_dataloader_builder=lambda epoch: dataloader,
        initial_key=jax.random.key(0),
        model=model,
        optimizer=optimizer,
        train_batch_fn=train_batch_fn,
        eval_batch_fn=eval_batch_fn,
        callbacks=callbacks,
        early_stopper=early_stopper,
    )


class TestEarlyStopper:
    def test_no_stop_while_improving(self) -> None:
        stopper = EarlyStopper(patience=3, min_delta=0.0)
        for loss in [1.0, 0.9, 0.8, 0.7]:
            assert stopper(jnp.array(loss)) is False

    def test_triggers_after_patience(self) -> None:
        stopper = EarlyStopper(patience=2, min_delta=0.0)
        assert stopper(jnp.array(1.0)) is False
        assert stopper(jnp.array(1.0)) is False
        assert stopper(jnp.array(1.0)) is True

    def test_min_delta(self) -> None:
        stopper = EarlyStopper(patience=1, min_delta=0.1)
        assert stopper(jnp.array(1.0)) is False
        assert stopper(jnp.array(0.95)) is True


class TestTrainLoop:
    def test_runs_for_n_epochs(self) -> None:
        recorder = ContextRecorder()
        _run_train_loop(
            n_epochs=5,
            dataloader=_make_dataloader(2),
            callbacks=(recorder,),
        )
        assert recorder.epochs == list(range(5))

    def test_tracks_best_val_loss(self) -> None:
        recorder = ContextRecorder()
        _run_train_loop(
            n_epochs=3,
            dataloader=_make_dataloader(1, value=0.5),
            callbacks=(recorder,),
        )
        assert recorder.best_val_losses == pytest.approx([0.5])

    def test_early_stopping(self) -> None:
        recorder = ContextRecorder()
        _run_train_loop(
            n_epochs=100,
            dataloader=_make_dataloader(1),
            callbacks=(recorder,),
            early_stopper=EarlyStopper(patience=2),
        )
        assert recorder.epochs == [0, 1, 2]

    def test_callback_phases(self) -> None:
        recorder = ContextRecorder()
        _run_train_loop(
            n_epochs=2,
            dataloader=_make_dataloader(1),
            callbacks=(recorder,),
        )
        assert recorder.phases == [
            TrainingPhase.INIT,
            TrainingPhase.EPOCH_START,
            TrainingPhase.EVAL_START,
            TrainingPhase.EPOCH_END,
            TrainingPhase.EPOCH_START,
            TrainingPhase.EVAL_START,
            TrainingPhase.EPOCH_END,
            TrainingPhase.BEFORE_RETURN,
        ]

    def test_graceful_keyboard_interrupt(self) -> None:
        recorder = ContextRecorder()
        call_count = 0

        def interrupting_step(
            model: nnx.Module,
            batch: Batch,
            keys: jax.Array,
            optimizer: nnx.Optimizer,
        ) -> jax.Array:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise KeyboardInterrupt
            return jnp.mean(batch[1])

        _run_train_loop(
            n_epochs=100,
            dataloader=_make_dataloader(1),
            train_batch_fn=interrupting_step,
            callbacks=(recorder,),
        )
        assert recorder.epochs == [0]
        assert recorder.phases[-1] == TrainingPhase.BEFORE_RETURN

    def test_with_batched_arrays(self) -> None:
        recorder = ContextRecorder()
        batch_size, features = 8, 10
        dataloader = [
            (
                jnp.ones((batch_size, features)),
                jnp.ones((batch_size, 1)),
            )
        ] * 3

        def shape_checking_step(
            model: nnx.Module,
            batch: Batch,
            keys: jax.Array,
            optimizer: nnx.Optimizer,
        ) -> jax.Array:
            assert batch[0].shape == (batch_size, features)
            assert keys.shape[0] == batch_size
            return jnp.mean(batch[1])

        _run_train_loop(
            n_epochs=2,
            dataloader=dataloader,
            train_batch_fn=shape_checking_step,
            eval_batch_fn=shape_checking_step,
            callbacks=(recorder,),
        )
        assert recorder.best_val_losses == pytest.approx([1.0])

    def test_print_every(self, capsys: pytest.CaptureFixture[str]) -> None:
        _run_train_loop(
            n_epochs=6,
            dataloader=_make_dataloader(1),
            callbacks=(PrintReporter(print_every=3),),
        )
        captured = capsys.readouterr()
        assert captured.out.count("Epoch") == 2
