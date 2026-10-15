"""The data of runs that consume a backbone run: the backbone's own windows.

A method run, a variance study or a posterior study restores a backbone run, so it
must see the data exactly as the backbone did: the same data-generation run (from
the backbone's ``data_generation_run_id`` tag) and the same fitted scalers (logged
into the backbone run), loaded, never refitted.
"""

from __future__ import annotations

import pathlib

from mlflow.tracking import MlflowClient

from qmodem.data import DataPipeline
from qmodem.tracking import load_fitted_scaler

from .data_processing import (
    WINDOW_SIZE,
    PreparedData,
    make_data_pipeline,
    prepare_data,
)

DATA_GENERATION_RUN_TAG = "data_generation_run_id"
"""Backbone-run tag naming the data-generation run it was trained on."""


def backbone_data_generation_run_id(
    backbone_run_id: str,
    *,
    expected: str | None = None,
    client: MlflowClient | None = None,
) -> str:
    """Return the data-generation run a backbone run was trained on.

    Args:
        backbone_run_id: The backbone run.
        expected: The configured data-generation run (e.g.
            ``DATA_GEN_RUN_ID_MULTI``), or ``None`` to accept the backbone's.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        The backbone's ``data_generation_run_id`` tag.

    Raises:
        ValueError: If ``expected`` is given and differs from the backbone's tag.
    """
    client = client if client is not None else MlflowClient()
    tagged = client.get_run(backbone_run_id).data.tags[DATA_GENERATION_RUN_TAG]
    if expected is not None and expected != tagged:
        raise ValueError(
            f"Backbone run {backbone_run_id} was trained on data-generation run "
            f"{tagged}, but {expected} is configured."
        )
    return tagged


def backbone_data_pipeline(
    backbone_run_id: str, window_size: int = WINDOW_SIZE
) -> DataPipeline:
    """Return the data pipeline with the backbone run's fitted scalers.

    Use it with ``fit_scalers=False`` (``prepare_windows`` / ``prepare_data``).

    Args:
        backbone_run_id: The backbone run holding ``x_scaler`` and ``y_scaler``.
        window_size: Number of time steps per window.

    Returns:
        The pipeline.
    """
    return make_data_pipeline(
        window_size,
        x_scaler=load_fitted_scaler(backbone_run_id, "x_scaler"),
        y_scaler=load_fitted_scaler(backbone_run_id, "y_scaler"),
    )


def prepare_backbone_data(
    backbone_run_id: str,
    *,
    raw_data_dir: pathlib.Path,
    expected_data_generation_run_id: str | None = None,
    client: MlflowClient | None = None,
) -> PreparedData:
    """Prepare the windows a backbone run was trained on, with its scalers.

    Args:
        backbone_run_id: The backbone run.
        raw_data_dir: Directory holding ``train.csv``.
        expected_data_generation_run_id: The configured data-generation run, or
            ``None``; see ``backbone_data_generation_run_id``.
        client: The MLflow client. Defaults to one on the current tracking URI.

    Returns:
        The backbone's train and validation windows.

    Raises:
        ValueError: If the configured data-generation run is not the backbone's.
    """
    data_run_id = backbone_data_generation_run_id(
        backbone_run_id, expected=expected_data_generation_run_id, client=client
    )
    return prepare_data(
        raw_data_dir,
        data_run_id,
        backbone_data_pipeline(backbone_run_id),
        fit_scalers=False,
    )
