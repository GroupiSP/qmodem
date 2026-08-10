from __future__ import annotations

import dataclasses
import pathlib

import jax
import jax.numpy as jnp
import mlflow
import numpy as np
import pandas as pd
from flax import nnx

from qmodem.battery.tracking import mlflow_load_scaler
from qmodem.metrics import point_crps


@dataclasses.dataclass
class HPOHyperparameters:
    """Hyperparameters for the HPO."""

    seed_objective: int = 42
    seed_hp_sampler: int = 123
    num_validation_histories: int = 5
    rul_grid_crps_start: float = 0.0
    rul_grid_crps_end: float = 5_000.0
    rul_grid_crps_resolution: float = 50.0
    num_mc_samples: int = 100
    num_hp_trials: int = 100
    eval_window_size: int = 250  # Number of time steps to use for evaluation of the model on validation histories


def pick_history_ids(num: int, seed: int, all_ids: list[int]) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.choice(all_ids, size=num, replace=False).tolist()


def _load_validation_history(path: pathlib.Path, id: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["run_id"] == id].sort_values("time")


def _get_eval_window(
    history: pd.DataFrame, features: list[str], eval_window_size: int
) -> np.ndarray:
    return history.iloc[:eval_window_size][features].to_numpy(dtype=np.float32)


def _get_true_rul_after_eval_window(
    history: pd.DataFrame, eval_window_size: int
) -> float:
    return np.float32(history.iloc[eval_window_size]["rul"])


def _scale_window(window: np.ndarray) -> np.ndarray:
    run_id = mlflow.active_run().info.run_id
    x_scaler = mlflow_load_scaler(f"runs:/{run_id}/x_scaler")
    return x_scaler.inverse_transform(window)


def _scale_ruls(ruls: np.ndarray) -> np.ndarray:
    run_id = mlflow.active_run().info.run_id
    y_scaler = mlflow_load_scaler(f"runs:/{run_id}/y_scaler")
    return y_scaler.inverse_transform(ruls)


def _build_rul_grid(start: float, end: float, resolution: int) -> np.ndarray:
    return np.arange(start, end, resolution)


def score_avg_val_crps(
    model: nnx.Module,
    hp: HPOHyperparameters,
    raw_data_dir: pathlib.Path,
    validation_history_ids: list[int],
    features: list[str] = ["voltage"],
) -> float:
    """Scores the model based on the point-CRPS metric computed for a few validation
    histories."""
    out = 0.0
    key = jax.random.key(hp.seed_objective)

    for history_id in pick_history_ids(
        num=hp.num_validation_histories,
        seed=hp.seed_objective + 1,
        all_ids=validation_history_ids,
    ):
        history = _load_validation_history(raw_data_dir / "train.csv", history_id)
        X_w = _get_eval_window(history, features, hp.eval_window_size)
        X_w = _scale_window(X_w)
        rul_true = _get_true_rul_after_eval_window(history, hp.eval_window_size)
        rul_grid = _build_rul_grid(
            start=hp.rul_grid_crps_start,
            end=hp.rul_grid_crps_end,
            resolution=hp.rul_grid_crps_resolution,
        )

        key, subkey = jax.random.split(key)
        samples_pred = model.mc_sample(
            subkey, jnp.array(X_w).reshape(1, -1, len(features)), hp.num_mc_samples
        )
        samples_pred = _scale_ruls(np.array(samples_pred, dtype=np.float32))

        crps_history = point_crps(rul_true, samples_pred, rul_grid)
        mlflow.log_metric(f"crps_history_{history_id}", crps_history)

        out += crps_history

    crps_avg = out / hp.num_validation_histories
    mlflow.log_metric("crps_avg", crps_avg)

    return crps_avg
