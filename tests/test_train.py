from __future__ import annotations

import jax.numpy as jnp
import pytest

from qmodem.train import EarlyStopper, train_loop

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataloader(n_batches: int, value: float = 1.0) -> list[jnp.ndarray]:
    """Return a simple list of scalar batches."""
    return [jnp.array(value)] * n_batches


def _identity_step(batch: jnp.ndarray) -> jnp.ndarray:
    """Step that returns the batch as a loss."""
    return batch


# ---------------------------------------------------------------------------
# EarlyStopper tests
# ---------------------------------------------------------------------------


class TestEarlyStopper:
    def test_no_stop_while_improving(self) -> None:
        stopper = EarlyStopper(patience=3, min_delta=0.0)
        for loss in [1.0, 0.9, 0.8, 0.7]:
            assert stopper(jnp.array(loss)) is False

    def test_triggers_after_patience(self) -> None:
        stopper = EarlyStopper(patience=2, min_delta=0.0)
        assert stopper(jnp.array(1.0)) is False  # improving
        assert stopper(jnp.array(1.0)) is False  # plateau, counter=1
        assert stopper(jnp.array(1.0)) is True  # plateau, counter=2 => stop

    def test_min_delta(self) -> None:
        stopper = EarlyStopper(patience=1, min_delta=0.1)
        assert stopper(jnp.array(1.0)) is False  # improving
        # Improvement < min_delta: treated as no improvement
        assert stopper(jnp.array(0.95)) is True


# ---------------------------------------------------------------------------
# train_loop tests
# ---------------------------------------------------------------------------


class TestTrainLoop:
    def test_runs_for_n_epochs(self) -> None:
        """train_loop completes exactly n_epochs epochs when no stopping occurs."""
        n_epochs = 5
        dl = _make_dataloader(2)
        _, epochs_completed = train_loop(
            n_epochs=n_epochs,
            dataloader_train=dl,
            dataloader_val=dl,
            train_batch_fn=_identity_step,
            eval_batch_fn=_identity_step,
        )
        assert epochs_completed == n_epochs

    def test_returns_best_val_loss(self) -> None:
        """The returned best_val_loss is the minimum validation loss seen."""
        dl = _make_dataloader(1, value=0.5)
        best, _ = train_loop(
            n_epochs=3,
            dataloader_train=dl,
            dataloader_val=dl,
            train_batch_fn=_identity_step,
            eval_batch_fn=_identity_step,
        )
        assert float(best) == pytest.approx(0.5)

    def test_early_stopping(self) -> None:
        """train_loop stops early when EarlyStopper fires."""
        stopper = EarlyStopper(patience=2, min_delta=0.0)
        # Constant loss => early stopper triggers after patience epochs
        dl = _make_dataloader(1, value=1.0)
        _, epochs_completed = train_loop(
            n_epochs=100,
            dataloader_train=dl,
            dataloader_val=dl,
            train_batch_fn=_identity_step,
            eval_batch_fn=_identity_step,
            early_stopper=stopper,
        )
        # patience=2: epoch1 (best), epoch2 (counter=1), epoch3 (counter=2 => stop)
        assert epochs_completed == 3

    def test_on_epoch_start_callbacks(self) -> None:
        """on_train_epoch_start and on_val_epoch_start are called each epoch."""
        train_calls: list[int] = []
        val_calls: list[int] = []

        dl = _make_dataloader(1)
        n_epochs = 4
        train_loop(
            n_epochs=n_epochs,
            dataloader_train=dl,
            dataloader_val=dl,
            train_batch_fn=_identity_step,
            eval_batch_fn=_identity_step,
            on_train_epoch_start=lambda: train_calls.append(1),
            on_val_epoch_start=lambda: val_calls.append(1),
        )
        assert len(train_calls) == n_epochs
        assert len(val_calls) == n_epochs

    def test_graceful_keyboard_interrupt(self) -> None:
        """A KeyboardInterrupt exits the loop and execution resumes."""
        interrupt_at_epoch = 2
        call_count = [0]

        def train_fn(batch: jnp.ndarray) -> jnp.ndarray:
            call_count[0] += 1
            if call_count[0] >= interrupt_at_epoch:
                raise KeyboardInterrupt
            return batch

        dl = _make_dataloader(1)
        # Should NOT raise; execution must resume after train_loop returns
        best, epochs_completed = train_loop(
            n_epochs=100,
            dataloader_train=dl,
            dataloader_val=dl,
            train_batch_fn=train_fn,
            eval_batch_fn=_identity_step,
        )
        assert epochs_completed < 100
        # best_val_loss should be a numeric value (either float or jax scalar)
        assert float(best) >= 0.0

    def test_with_batched_arrays(self) -> None:
        """train_loop works correctly with multi-dimensional batches."""
        batch_size, features = 8, 10
        dl = [jnp.ones((batch_size, features))] * 3

        def train_fn(batch: jnp.ndarray) -> jnp.ndarray:
            assert batch.shape == (batch_size, features)
            return jnp.zeros((batch_size, features))

        def eval_fn(batch: jnp.ndarray) -> jnp.ndarray:
            return jnp.mean(batch)

        best, epochs = train_loop(
            n_epochs=2,
            dataloader_train=dl,
            dataloader_val=dl,
            train_batch_fn=train_fn,
            eval_batch_fn=eval_fn,
        )
        assert epochs == 2
        assert float(best) == pytest.approx(1.0)

    def test_print_every(self, capsys: pytest.CaptureFixture) -> None:
        """Progress is printed on epoch 1 and every print_every epochs."""
        dl = _make_dataloader(1)
        n_epochs = 6
        print_every = 3
        train_loop(
            n_epochs=n_epochs,
            dataloader_train=dl,
            dataloader_val=dl,
            train_batch_fn=_identity_step,
            eval_batch_fn=_identity_step,
            print_every=print_every,
        )
        captured = capsys.readouterr()
        # Epochs printed: 1, 3, 6 → 3 "Epoch" lines
        assert captured.out.count("Epoch") == 3
