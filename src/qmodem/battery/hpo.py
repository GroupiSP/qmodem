from __future__ import annotations

import dataclasses
import pathlib

import flax.nnx as nnx
import jax
import mlflow
import numpy as np
import pandas as pd

from qmodem.metrics import point_crps
from qmodem.tracking import MLFlowSetup, track_mlflow


@dataclasses.dataclass
class HPOHyperparameters:
    """Hyperparameters for the HPO objective function."""

    seed: int = 42
    num_validation_histories: int = 5
    rul_grid_crps_start: float = 0.0
    rul_grid_crps_end: float = 12_000.0
    rul_grid_crps_resolution: int = 60
    num_mc_samples: int = 100


def pick_history_ids(num: int, seed: int, all_ids: list[int]) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.choice(all_ids, size=num, replace=False).tolist()


def _load_validation_history(path: pathlib.Path, id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["run_id"] == id].sort_values("time")


def _get_first_time_window(history: pd.DataFrame, window_size: int) -> np.ndarray:
    return history.iloc[:window_size]["voltage"].to_numpy()


def _get_true_rul_after_first_window(history: pd.DataFrame, window_size: int) -> float:
    t = history.iloc[window_size]["time"].to_numpy()
    t_eod = history.iloc[-1]["time"].to_numpy()
    return t_eod - t


def _build_rul_grid(start: float, end: float, resolution: int) -> np.ndarray:
    return np.arange(start, end, resolution)


def score_avg_val_crps(
    model: nnx.Module,
    mlflow_setup: MLFlowSetup,
    hp: HPOHyperparameters,
    raw_data_dir: pathlib.Path,
    validation_history_ids: list[int],
    window_size: int,
) -> float:
    """Scores the model based on the point-CRPS metric computed for a few validation
    histories."""
    out = 0.0
    key = jax.random.key(hp.seed)

    with track_mlflow(setup=mlflow_setup):
        for history_id in pick_history_ids(
            num=hp.num_validation_histories,
            seed=hp.seed + 1,
            all_ids=validation_history_ids,
        ):
            history = _load_validation_history(raw_data_dir / "train.csv", history_id)
            X_w = _get_first_time_window(history, window_size)
            rul_true = _get_true_rul_after_first_window(history, window_size)
            rul_grid = _build_rul_grid(
                start=hp.rul_grid_crps_start,
                end=hp.rul_grid_crps_end,
                resolution=hp.rul_grid_crps_resolution,
            )

            key, subkey = jax.random.split(key)
            samples_pred = model.mc_sample(subkey, X_w, hp.num_mc_samples)

            crps_history = point_crps(rul_true, samples_pred, rul_grid)
            mlflow.log_metric(f"crps_history_{history_id}", crps_history)

            out += crps_history

        crps_avg = out / hp.num_validation_histories
        mlflow.log_metric("crps_avg", crps_avg)

    return crps_avg
