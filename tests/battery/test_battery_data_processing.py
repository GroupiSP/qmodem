"""Tests for the battery loader adapter over the prepared windows."""

from __future__ import annotations

import dataclasses
import math

import jax
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.data_generation import Hyperparameters
from qmodem.battery.data_processing import (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    WINDOW_SIZE,
    HistorySplits,
    PreparedData,
    SplitSizes,
    ValidationSet,
    make_data_pipeline,
    make_loader,
    posterior_dataset_size,
    prepare_windows,
    split_histories,
)
from qmodem.data import ArrayDataSource

seeds = st.integers(min_value=0, max_value=2**31 - 1)
epochs = st.integers(min_value=0, max_value=10_000)
batch_sizes = st.integers(min_value=1, max_value=64)
validation_sets = st.sampled_from(ValidationSet)


def _stack(batches: list, field: str) -> np.ndarray:
    return np.concatenate([np.asarray(getattr(b, field)) for b in batches])


@settings(deadline=None, max_examples=25)
@given(seed=seeds, epoch=epochs, batch_size=batch_sizes)
def test_same_epoch_gives_the_same_batch_order(
    synthetic_battery, seed: int, epoch: int, batch_size: int
) -> None:
    loader = make_loader(
        synthetic_battery.data,
        batch_size=batch_size,
        key=jax.random.key(seed),
        validation=ValidationSet.BACKBONE,
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
    loader = make_loader(
        data,
        batch_size=batch_size,
        key=jax.random.key(seed),
        validation=ValidationSet.BACKBONE,
    )

    batches = loader(split="train", epoch=epoch)

    assert all(len(b.targets) <= batch_size for b in batches)
    inputs = _stack(batches, "inputs")
    expected = np.asarray(data.train.features)
    order = np.lexsort(inputs.reshape(len(inputs), -1).T)
    expected_order = np.lexsort(expected.reshape(len(expected), -1).T)
    np.testing.assert_array_equal(inputs[order], expected[expected_order])


@settings(deadline=None, max_examples=25)
@given(seed=seeds, epoch=epochs, batch_size=batch_sizes, validation=validation_sets)
def test_validation_split_is_the_requested_validation_windows_in_order(
    synthetic_battery,
    seed: int,
    epoch: int,
    batch_size: int,
    validation: ValidationSet,
) -> None:
    data = synthetic_battery.data
    loader = make_loader(
        data, batch_size=batch_size, key=jax.random.key(seed), validation=validation
    )
    expected = {
        ValidationSet.BACKBONE: data.backbone_val,
        ValidationSet.POSTERIOR: data.posterior_val,
    }[validation]

    batches = loader(split="validation", epoch=epoch)

    np.testing.assert_array_equal(
        _stack(batches, "inputs"), np.asarray(expected.features)
    )
    np.testing.assert_array_equal(
        _stack(batches, "targets"), np.asarray(expected.targets)
    )


@settings(deadline=None, max_examples=25)
@given(seed=seeds, epoch=epochs, batch_size=batch_sizes)
def test_the_train_split_is_the_same_whichever_validation_set_is_served(
    synthetic_battery, seed: int, epoch: int, batch_size: int
) -> None:
    batches = [
        make_loader(
            synthetic_battery.data,
            batch_size=batch_size,
            key=jax.random.key(seed),
            validation=validation,
        )(split="train", epoch=epoch)
        for validation in ValidationSet
    ]

    for field in ("inputs", "targets"):
        np.testing.assert_array_equal(
            _stack(batches[0], field), _stack(batches[1], field)
        )


def test_the_validation_sets_have_their_histories_windows(synthetic_battery) -> None:
    # Stride 1: a history of length L gives L - W + 1 windows.
    data = synthetic_battery.data

    assert len(data.backbone_val) == sum(
        length - WINDOW_SIZE + 1 for length in synthetic_battery.backbone_val_lengths
    )
    assert len(data.posterior_val) == sum(
        length - WINDOW_SIZE + 1 for length in synthetic_battery.posterior_val_lengths
    )


def test_the_window_counts_name_every_split_and_k(synthetic_battery) -> None:
    # Train (23, 31, 27, 40), backbone (25, 19), posterior (21, 30, 17); W = 10.
    # Every validation history has fewer than K = 50 windows: all are kept.
    assert synthetic_battery.data.window_counts() == {
        "windows_train": 85,
        "windows_backbone_val": 26,
        "windows_posterior_val": 41,
        "validation_timestamps": 50,
    }


def test_the_window_counts_reflect_the_validation_timestamps(
    synthetic_histories,
) -> None:
    # Windows per history: backbone (16, 10), posterior (12, 21, 8); K = 9.
    data = _prepare(synthetic_histories, [25, 19], [21, 30, 17], k=9)

    assert data.window_counts() == {
        "windows_train": 85,
        "windows_backbone_val": 9 + 9,
        "windows_posterior_val": 9 + 9 + 8,
        "validation_timestamps": 9,
    }


def test_batches_have_window_inputs_and_column_targets(synthetic_battery) -> None:
    loader = make_loader(
        synthetic_battery.data,
        batch_size=8,
        key=jax.random.key(0),
        validation=ValidationSet.BACKBONE,
    )

    for split in ("train", "validation"):
        for batch in loader(split=split, epoch=0):
            assert batch.inputs.shape[1:] == (WINDOW_SIZE, 2)
            assert batch.targets.shape == (batch.inputs.shape[0], 1)


def test_the_shuffle_depends_on_the_seed(synthetic_battery) -> None:
    data = synthetic_battery.data
    orders = [
        _stack(
            make_loader(
                data,
                batch_size=8,
                key=jax.random.key(s),
                validation=ValidationSet.BACKBONE,
            )(split="train", epoch=3),
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
    loader = make_loader(
        synthetic_battery.data,
        batch_size=7,
        key=jax.random.key(0),
        validation=ValidationSet.BACKBONE,
    )

    assert posterior_dataset_size(synthetic_battery.data) == expected
    assert sum(len(b.targets) for b in loader(split="train", epoch=0)) == expected


@settings(deadline=None, max_examples=25)
@given(seed=seeds, epoch=epochs, batch_size=batch_sizes)
def test_train_batches_are_full_except_the_last(
    synthetic_battery, seed: int, epoch: int, batch_size: int
) -> None:
    n_windows = len(synthetic_battery.data.train)
    loader = make_loader(
        synthetic_battery.data,
        batch_size=batch_size,
        key=jax.random.key(seed),
        validation=ValidationSet.BACKBONE,
    )

    sizes = [len(b.targets) for b in loader(split="train", epoch=epoch)]

    assert len(sizes) == math.ceil(n_windows / batch_size)
    assert all(size == batch_size for size in sizes[:-1])
    assert 0 < sizes[-1] <= batch_size


@settings(deadline=None, max_examples=25)
@given(split=st.text().filter(lambda s: s not in (TRAIN_SPLIT, VALIDATION_SPLIT)))
def test_an_unknown_split_is_rejected(synthetic_battery, split: str) -> None:
    loader = make_loader(
        synthetic_battery.data,
        batch_size=8,
        key=jax.random.key(0),
        validation=ValidationSet.BACKBONE,
    )

    with pytest.raises(ValueError, match="Unknown split"):
        loader(split=split, epoch=0)


# --- The three-way split of train.csv by run_id ----------------------------------

split_sizes = st.builds(
    SplitSizes,
    n_train=st.integers(0, 6),
    n_backbone_val=st.integers(0, 6),
    n_posterior_val=st.integers(0, 6),
)
row_counts = st.lists(st.integers(1, 3), min_size=30, max_size=30)


def _total(sizes: SplitSizes) -> int:
    return sizes.n_train + sizes.n_backbone_val + sizes.n_posterior_val


def _histories_frame(n_histories: int, rows: list[int], seed: int) -> pd.DataFrame:
    run_ids = np.repeat(np.arange(n_histories), rows[:n_histories])
    order = np.random.default_rng(seed).permutation(len(run_ids))
    return pd.DataFrame({"run_id": run_ids[order], "time": np.arange(len(run_ids))})


def _run_ids(frame: pd.DataFrame) -> set[int]:
    return set(frame["run_id"].unique().tolist())


@settings(deadline=None, max_examples=50)
@given(sizes=split_sizes, rows=row_counts, seed=seeds)
def test_the_validation_sets_are_disjoint_and_cover_the_histories_with_train(
    sizes: SplitSizes, rows: list[int], seed: int
) -> None:
    histories = _histories_frame(_total(sizes), rows, seed)

    splits = split_histories(histories, sizes)

    train = _run_ids(splits.train)
    backbone, posterior = _run_ids(splits.backbone_val), _run_ids(splits.posterior_val)
    assert backbone.isdisjoint(posterior)
    assert train.isdisjoint(backbone | posterior)
    assert train | backbone | posterior == set(range(_total(sizes)))
    frames = (splits.train, splits.backbone_val, splits.posterior_val)
    assert sum(len(frame) for frame in frames) == len(histories)


@settings(deadline=None, max_examples=25)
@given(sizes=split_sizes, rows=row_counts, seed=seeds)
def test_the_splits_are_contiguous_run_id_blocks_of_the_logged_sizes_in_order(
    sizes: SplitSizes, rows: list[int], seed: int
) -> None:
    splits = split_histories(_histories_frame(_total(sizes), rows, seed), sizes)

    start = 0
    for frame, size in zip(
        (splits.train, splits.backbone_val, splits.posterior_val),
        (sizes.n_train, sizes.n_backbone_val, sizes.n_posterior_val),
        strict=True,
    ):
        assert _run_ids(frame) == set(range(start, start + size))
        start += size


@settings(deadline=None, max_examples=25)
@given(sizes=split_sizes, offset=st.integers(-3, 3).filter(bool), seed=seeds)
def test_histories_that_do_not_match_the_logged_sizes_are_refused(
    sizes: SplitSizes, offset: int, seed: int
) -> None:
    n_histories = max(_total(sizes) + offset, 0)
    if n_histories == _total(sizes):
        return

    with pytest.raises(ValueError, match="run_id"):
        split_histories(_histories_frame(n_histories, [1] * 30, seed), sizes)


def test_split_sizes_are_read_from_the_data_generation_params() -> None:
    params = {
        "n_histories_train": "10",
        "n_histories_backbone_val": "20",
        "n_histories_posterior_val": "20",
        "n_histories_test": "20",
    }

    assert SplitSizes.from_params(params) == SplitSizes(
        n_train=10, n_backbone_val=20, n_posterior_val=20
    )


@settings(deadline=None, max_examples=25)
@given(sizes=split_sizes)
def test_split_sizes_round_trip_through_the_logged_data_generation_params(
    sizes: SplitSizes,
) -> None:
    hp = Hyperparameters(
        n_histories_train=sizes.n_train,
        n_policy_1_train=0,
        n_histories_backbone_val=sizes.n_backbone_val,
        n_policy_1_backbone_val=0,
        n_histories_posterior_val=sizes.n_posterior_val,
        n_policy_1_posterior_val=0,
    )
    logged = {key: str(value) for key, value in dataclasses.asdict(hp).items()}

    assert SplitSizes.from_params(logged) == sizes


# --- The validation timestamps of both validation sets ----------------------------

ALL_WINDOWS = 10**6
"""A K above any history's window count: every stride-1 window is kept."""

TRAIN_LENGTHS: tuple[int, ...] = (23, 31, 27, 40)

history_lengths = st.lists(st.integers(3, 60), min_size=1, max_size=4)
validation_timestamp_counts = st.integers(2, 40)


def _prepare(
    synthetic_histories,
    backbone_lengths: list[int],
    posterior_lengths: list[int],
    k: int,
) -> PreparedData:
    splits = HistorySplits(
        train=synthetic_histories(TRAIN_LENGTHS, seed=0),
        backbone_val=synthetic_histories(tuple(backbone_lengths), seed=1),
        posterior_val=synthetic_histories(tuple(posterior_lengths), seed=2),
    )
    return prepare_windows(splits, make_data_pipeline(), validation_timestamps=k)


def _windows_per_history(lengths: list[int]) -> list[int]:
    # Stride 1; a history shorter than a window is padded into one window.
    return [max(length - WINDOW_SIZE + 1, 1) for length in lengths]


def _selected_positions(
    reduced: ArrayDataSource, full: ArrayDataSource, lengths: list[int]
) -> list[list[int]]:
    """Locate each reduced window among the stride-1 windows of its history.

    Returns:
        Per history, the stride-1 index of each of its reduced windows, in the
        reduced set's order.
    """
    full_flat = np.asarray(full.features).reshape(len(full), -1)
    reduced_flat = np.asarray(reduced.features).reshape(len(reduced), -1)
    distances = np.abs(reduced_flat[:, None, :] - full_flat[None, :, :]).max(-1)
    matches = distances.argmin(axis=1)
    np.testing.assert_allclose(distances.min(axis=1), 0.0, atol=1e-6)
    starts = np.cumsum([0, *_windows_per_history(lengths)])
    history = np.searchsorted(starts, matches, side="right") - 1
    return [
        [int(m - starts[h]) for m, hh in zip(matches, history) if hh == h]
        for h in range(len(lengths))
    ]


@settings(deadline=None, max_examples=25)
@given(
    backbone=history_lengths, posterior=history_lengths, k=validation_timestamp_counts
)
def test_every_validation_history_contributes_min_k_and_its_window_count(
    synthetic_histories, backbone: list[int], posterior: list[int], k: int
) -> None:
    data = _prepare(synthetic_histories, backbone, posterior, k)
    full = _prepare(synthetic_histories, backbone, posterior, ALL_WINDOWS)

    for which, lengths in (
        (ValidationSet.BACKBONE, backbone),
        (ValidationSet.POSTERIOR, posterior),
    ):
        assert len(full.validation(which)) == sum(_windows_per_history(lengths))
        positions = _selected_positions(
            data.validation(which), full.validation(which), lengths
        )
        assert [len(p) for p in positions] == [
            min(k, n) for n in _windows_per_history(lengths)
        ]


@settings(deadline=None, max_examples=25)
@given(
    backbone=history_lengths, posterior=history_lengths, k=validation_timestamp_counts
)
def test_validation_windows_are_evenly_spaced_stride_1_windows_in_order_with_both_ends(
    synthetic_histories, backbone: list[int], posterior: list[int], k: int
) -> None:
    data = _prepare(synthetic_histories, backbone, posterior, k)
    full = _prepare(synthetic_histories, backbone, posterior, ALL_WINDOWS)

    for which, lengths in (
        (ValidationSet.BACKBONE, backbone),
        (ValidationSet.POSTERIOR, posterior),
    ):
        positions = _selected_positions(
            data.validation(which), full.validation(which), lengths
        )
        for selected, n in zip(positions, _windows_per_history(lengths), strict=True):
            assert selected[0] == 0
            assert selected[-1] == n - 1
            gaps = np.diff(selected)
            assert np.all(gaps >= 1)  # distinct and in order
            if len(gaps):
                assert gaps.max() - gaps.min() <= 1  # evenly spaced


@settings(deadline=None, max_examples=15)
@given(
    backbone=history_lengths, posterior=history_lengths, k=validation_timestamp_counts
)
def test_the_validation_timestamps_are_deterministic(
    synthetic_histories, backbone: list[int], posterior: list[int], k: int
) -> None:
    first = _prepare(synthetic_histories, backbone, posterior, k)
    second = _prepare(synthetic_histories, backbone, posterior, k)

    for which in ValidationSet:
        for field in ("features", "targets"):
            np.testing.assert_array_equal(
                np.asarray(getattr(first.validation(which), field)),
                np.asarray(getattr(second.validation(which), field)),
            )


@settings(deadline=None, max_examples=15)
@given(
    backbone=history_lengths,
    posterior=history_lengths,
    k=validation_timestamp_counts,
    other_k=validation_timestamp_counts,
)
def test_the_training_windows_are_the_same_for_any_k(
    synthetic_histories,
    backbone: list[int],
    posterior: list[int],
    k: int,
    other_k: int,
) -> None:
    trains = [
        _prepare(synthetic_histories, backbone, posterior, kk).train
        for kk in (k, other_k, ALL_WINDOWS)
    ]

    for train in trains[1:]:
        for field in ("features", "targets"):
            np.testing.assert_array_equal(
                np.asarray(getattr(train, field)), np.asarray(getattr(trains[0], field))
            )


@settings(deadline=None, max_examples=15)
@given(
    backbone=history_lengths, posterior=history_lengths, k=validation_timestamp_counts
)
def test_the_validation_sets_stay_disjoint(
    synthetic_histories, backbone: list[int], posterior: list[int], k: int
) -> None:
    data = _prepare(synthetic_histories, backbone, posterior, k)
    full = _prepare(synthetic_histories, backbone, posterior, ALL_WINDOWS)

    # Every backbone window is one of the backbone set's own windows, and none is
    # a posterior validation set window (and vice versa).
    for which, other in (
        (ValidationSet.BACKBONE, ValidationSet.POSTERIOR),
        (ValidationSet.POSTERIOR, ValidationSet.BACKBONE),
    ):
        own = np.asarray(full.validation(which).features).reshape(
            len(full.validation(which)), -1
        )
        foreign = np.asarray(full.validation(other).features).reshape(
            len(full.validation(other)), -1
        )
        for window in np.asarray(data.validation(which).features).reshape(
            len(data.validation(which)), -1
        ):
            assert np.abs(own - window).max(-1).min() < 1e-6
            assert np.abs(foreign - window).max(-1).min() > 1e-6


@pytest.mark.parametrize("k", [-1, 0, 1])
def test_fewer_than_two_validation_timestamps_are_refused(
    synthetic_histories, k: int
) -> None:
    with pytest.raises(ValueError, match="K must be at least 2"):
        _prepare(synthetic_histories, [25], [25], k)
