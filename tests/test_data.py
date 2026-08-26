"""Tests for qmodem.data module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qmodem.data import _make_windows, get_time_windows_and_join


def _descending_ruls(N_t: int) -> np.ndarray:
    return np.arange(N_t - 1, -1, -1, dtype=float)


def _make_run_df(
    run_lengths: dict[str | int, int], feature_cols: list[str]
) -> pd.DataFrame:
    """Build a minimal dataframe for get_time_windows_and_join with integer time
    steps."""
    rows = []
    for run_id, N_t in run_lengths.items():
        for t in range(N_t):
            row: dict = {"run_id": run_id, "time": float(t), "rul": float(N_t - 1 - t)}
            for col in feature_cols:
                row[col] = float(t)
            rows.append(row)
    return pd.DataFrame(rows)


def test_make_windows_works_with_1d_features():
    features = np.arange(6, dtype=float)  # shape (6,)
    ruls = _descending_ruls(6)
    windows, targets = _make_windows(features, ruls, window_size=3, stride=1)
    assert all(w.shape == (3,) for w in windows)


def test_make_windows_works_with_multidimensional_features():
    features = np.arange(12, dtype=float).reshape(6, 2)  # shape (6, 2)
    ruls = _descending_ruls(6)
    windows, targets = _make_windows(features, ruls, window_size=3, stride=1)
    assert all(w.shape == (3, 2) for w in windows)


def test_make_windows_last_target_is_zero():
    features = np.arange(12, dtype=float).reshape(6, 2)
    ruls = _descending_ruls(6)
    _, targets = _make_windows(features, ruls, window_size=3, stride=1)
    assert targets[-1] == 0.0


def test_make_windows_alignment_target_is_rul_at_window_end():
    # For the window starting at `start`, target = ruls[start + window_size]
    N_t, window_size, stride = 6, 3, 1
    features = np.arange(N_t * 2, dtype=float).reshape(N_t, 2)
    ruls = _descending_ruls(N_t)  # [5, 4, 3, 2, 1, 0]
    windows, targets = _make_windows(features, ruls, window_size, stride)
    assert targets[0] == ruls[window_size]
    assert targets[1] == ruls[stride + window_size]
    assert targets[2] == ruls[2 * stride + window_size]


def test_make_windows_short_history_padded_with_first_value():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])  # N_t=2 < window_size=5
    ruls = np.array([1.0, 0.0])
    windows, targets = _make_windows(features, ruls, window_size=5, stride=1)
    assert len(windows) >= 1
    pad_len = 5 - 2
    np.testing.assert_array_equal(
        windows[0][:pad_len],
        np.full((pad_len, 2), features[0]),
    )


def test_make_windows_window_shape_and_scalar_targets():
    N_t, window_size, N_i = 10, 4, 3
    features = np.random.rand(N_t, N_i)
    ruls = _descending_ruls(N_t)
    windows, targets = _make_windows(features, ruls, window_size, stride=1)
    for w in windows:
        assert w.shape == (window_size, N_i)
    for t in targets:
        assert isinstance(t, float)


@pytest.mark.parametrize(
    "N_t,window_size,stride",
    [
        (10, 4, 2),  # (N_t - window_size) divisible by stride
        (9, 3, 4),  # (N_t - window_size) NOT divisible by stride
        (7, 3, 1),
    ],
)
def test_make_windows_number_of_windows(N_t, window_size, stride):
    features = np.random.rand(N_t, 2)
    ruls = _descending_ruls(N_t)
    windows, targets = _make_windows(features, ruls, window_size, stride)
    expected = int(np.ceil((N_t - window_size) / stride)) + 1
    assert len(windows) == expected
    assert len(targets) == expected


def test_get_time_windows_and_join_output_shapes_1d_features():
    window_size, stride = 3, 1
    df = _make_run_df({0: 6, 1: 8}, feature_cols=["feat_a"])
    X, y = get_time_windows_and_join(df, window_size, stride, features=["feat_a"])
    assert X.ndim == 3
    assert X.shape[1] == window_size
    assert X.shape[2] == 1
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]


def test_get_time_windows_and_join_output_shapes_multid_features():
    window_size, stride, N_i = 3, 1, 2
    df = _make_run_df({0: 6, 1: 8}, feature_cols=["feat_a", "feat_b"])
    X, y = get_time_windows_and_join(
        df, window_size, stride, features=["feat_a", "feat_b"]
    )
    assert X.shape[1:] == (window_size, N_i)
    assert y.shape == (X.shape[0],)


def test_get_time_windows_and_join_single_unit():
    window_size, stride = 4, 2
    df = _make_run_df({0: 10}, feature_cols=["feat_a"])
    X, y = get_time_windows_and_join(df, window_size, stride, features=["feat_a"])
    assert X.shape[0] == y.shape[0]
    assert X.shape[1:] == (window_size, 1)


@pytest.mark.parametrize(
    "run_lengths,window_size,stride",
    [
        ({0: 9, 1: 6}, 3, 2),
        ({0: 10, 1: 10, 2: 10}, 4, 2),
        ({0: 7}, 3, 1),
    ],
)
def test_get_time_windows_and_join_total_window_count(run_lengths, window_size, stride):
    feature_cols = ["feat_a"]
    df = _make_run_df(run_lengths, feature_cols)
    X, y = get_time_windows_and_join(df, window_size, stride, features=feature_cols)
    expected = sum(
        int(np.ceil((N_t - window_size) / stride)) + 1 for N_t in run_lengths.values()
    )
    assert len(X) == expected
    assert len(y) == expected
