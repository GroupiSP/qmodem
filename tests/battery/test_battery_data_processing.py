"""Tests for the battery loader adapter over the prepared windows."""

from __future__ import annotations

import math

import jax
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.data_processing import (
    WINDOW_SIZE,
    make_loader,
    posterior_dataset_size,
)

seeds = st.integers(min_value=0, max_value=2**31 - 1)
epochs = st.integers(min_value=0, max_value=10_000)
batch_sizes = st.integers(min_value=1, max_value=64)


def _stack(batches: list, field: str) -> np.ndarray:
    return np.concatenate([np.asarray(getattr(b, field)) for b in batches])


@settings(deadline=None, max_examples=25)
@given(seed=seeds, epoch=epochs, batch_size=batch_sizes)
def test_same_epoch_gives_the_same_batch_order(
    synthetic_battery, seed: int, epoch: int, batch_size: int
) -> None:
    loader = make_loader(
        synthetic_battery.data, batch_size=batch_size, key=jax.random.key(seed)
    )

    first = loader(split="train", epoch=epoch)
    second = loader(split="train", epoch=epoch)

    np.testing.assert_array_equal(_stack(first, "inputs"), _stack(second, "inputs"))
    np.testing.assert_array_equal(_stack(first, "targets"), _stack(second, "targets"))


@settings(deadline=None, max_examples=25)
@given(seed=seeds, epoch=epochs, batch_size=batch_sizes)
def test_train_split_is_a_permutation_of_the_training_windows(
    synthetic_battery, seed: int, epoch: int, batch_size: int
) -> None:
    data = synthetic_battery.data
    loader = make_loader(data, batch_size=batch_size, key=jax.random.key(seed))

    batches = loader(split="train", epoch=epoch)

    assert all(len(b.targets) <= batch_size for b in batches)
    inputs = _stack(batches, "inputs")
    expected = np.asarray(data.train.features)
    order = np.lexsort(inputs.reshape(len(inputs), -1).T)
    expected_order = np.lexsort(expected.reshape(len(expected), -1).T)
    np.testing.assert_array_equal(inputs[order], expected[expected_order])


@settings(deadline=None, max_examples=25)
@given(seed=seeds, epoch=epochs, batch_size=batch_sizes)
def test_validation_split_is_the_validation_windows_in_order(
    synthetic_battery, seed: int, epoch: int, batch_size: int
) -> None:
    data = synthetic_battery.data
    loader = make_loader(data, batch_size=batch_size, key=jax.random.key(seed))

    batches = loader(split="validation", epoch=epoch)

    np.testing.assert_array_equal(
        _stack(batches, "inputs"), np.asarray(data.val.features)
    )
    np.testing.assert_array_equal(
        _stack(batches, "targets"), np.asarray(data.val.targets)
    )


def test_batches_have_window_inputs_and_column_targets(synthetic_battery) -> None:
    loader = make_loader(synthetic_battery.data, batch_size=8, key=jax.random.key(0))

    for split in ("train", "validation"):
        for batch in loader(split=split, epoch=0):
            assert batch.inputs.shape[1:] == (WINDOW_SIZE, 2)
            assert batch.targets.shape == (batch.inputs.shape[0], 1)


def test_the_shuffle_depends_on_the_seed(synthetic_battery) -> None:
    data = synthetic_battery.data
    orders = [
        _stack(
            make_loader(data, batch_size=8, key=jax.random.key(s))(
                split="train", epoch=3
            ),
            "targets",
        )
        for s in (0, 1)
    ]

    assert not np.array_equal(orders[0], orders[1])


def test_posterior_dataset_size_is_the_number_of_training_windows(
    synthetic_battery,
) -> None:
    # Stride 1: a history of length L gives L - W + 1 windows.
    expected = sum(
        length - WINDOW_SIZE + 1 for length in synthetic_battery.train_lengths
    )
    loader = make_loader(synthetic_battery.data, batch_size=7, key=jax.random.key(0))

    assert posterior_dataset_size(synthetic_battery.data) == expected
    assert sum(len(b.targets) for b in loader(split="train", epoch=0)) == expected


@settings(deadline=None, max_examples=25)
@given(seed=seeds, epoch=epochs, batch_size=batch_sizes)
def test_train_batches_are_full_except_the_last(
    synthetic_battery, seed: int, epoch: int, batch_size: int
) -> None:
    n_windows = len(synthetic_battery.data.train)
    loader = make_loader(
        synthetic_battery.data, batch_size=batch_size, key=jax.random.key(seed)
    )

    sizes = [len(b.targets) for b in loader(split="train", epoch=epoch)]

    assert len(sizes) == math.ceil(n_windows / batch_size)
    assert all(size == batch_size for size in sizes[:-1])
    assert 0 < sizes[-1] <= batch_size
