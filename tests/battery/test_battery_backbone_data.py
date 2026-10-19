"""Runs that consume a backbone run window the data exactly as the backbone did."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import sklearn.preprocessing as skpp
from hypothesis import given, settings
from hypothesis import strategies as st

from qmodem.battery.backbone_data import (
    backbone_data_generation_run_id,
    backbone_data_pipeline,
)
from qmodem.battery.data_processing import (
    FEATURES,
    WINDOW_SIZE,
    HistorySplits,
    make_data_pipeline,
    prepare_windows,
)
from qmodem.battery.train import track_backbone_run
from qmodem.tracking import ScalerArtifact

DATA_RUN_ID = "data-generation-run"


def _histories(lengths: tuple[int, ...], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for run_id, length in enumerate(lengths):
        time = 5.0 * np.arange(length)
        frames.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "time": time,
                    "load": 1.0 + 0.1 * run_id + 0.01 * rng.standard_normal(length),
                    "voltage": np.linspace(4.2, 3.4, length),
                    "rul": time[-1] - time,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def tracked_backbone(mlflow_backend, trained_backbone):
    """A backbone run whose scalers were fitted on other data than the tests use."""
    rng = np.random.default_rng(0)
    x_scaler = skpp.StandardScaler().fit(rng.normal(3.0, 2.0, (50, len(FEATURES))))
    y_scaler = skpp.MinMaxScaler().fit(rng.uniform(0.0, 900.0, (50, 1)))
    backbone = track_backbone_run(
        data=trained_backbone.battery.data,
        mean_hp=trained_backbone.mean_hp,
        variance_hp=trained_backbone.variance_hp,
        seed=trained_backbone.seed,
        data_generation_run_id=DATA_RUN_ID,
        scalers=(
            ScalerArtifact("x_scaler", x_scaler, np.ones((WINDOW_SIZE, 2))),
            ScalerArtifact("y_scaler", y_scaler, np.ones((1, 1)), "inverse_transform"),
        ),
        setup=mlflow_backend.setup,
    )
    return backbone.run_id, x_scaler, y_scaler


def test_the_data_run_comes_from_the_backbone_tag(
    tracked_backbone, mlflow_backend
) -> None:
    run_id, _, _ = tracked_backbone
    client = mlflow_backend.client

    assert backbone_data_generation_run_id(run_id, client=client) == DATA_RUN_ID
    assert (
        backbone_data_generation_run_id(run_id, expected=DATA_RUN_ID, client=client)
        == DATA_RUN_ID
    )


def test_a_configured_data_run_that_disagrees_with_the_backbone_is_refused(
    tracked_backbone, mlflow_backend
) -> None:
    run_id, _, _ = tracked_backbone

    with pytest.raises(ValueError, match="data-generation"):
        backbone_data_generation_run_id(
            run_id, expected="another-data-run", client=mlflow_backend.client
        )


@settings(max_examples=5, deadline=None)
@given(seed=st.integers(0, 2**16))
def test_windows_are_scaled_with_the_backbone_scalers_never_refitted(
    tracked_backbone, seed: int
) -> None:
    run_id, x_scaler, y_scaler = tracked_backbone
    splits = HistorySplits(
        train=_histories((23, 31), seed),
        backbone_val=_histories((19,), seed + 1),
        posterior_val=_histories((17, 22), seed + 2),
    )

    loaded = prepare_windows(splits, backbone_data_pipeline(run_id), fit_scalers=False)
    parent = prepare_windows(
        splits,
        make_data_pipeline(
            WINDOW_SIZE,
            x_scaler=copy.deepcopy(x_scaler),
            y_scaler=copy.deepcopy(y_scaler),
        ),
        fit_scalers=False,
    )

    for split in ("train", "backbone_val", "posterior_val"):
        np.testing.assert_array_equal(
            getattr(loaded, split).features, getattr(parent, split).features
        )
        np.testing.assert_array_equal(
            getattr(loaded, split).targets, getattr(parent, split).targets
        )
