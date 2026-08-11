from __future__ import annotations

import dataclasses
import pathlib
import pickle
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from enum import StrEnum, auto
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import simbat as sb


class VOCModel(StrEnum):
    BUSTOS_BAEZA = auto()


class ECMModel(StrEnum):
    THEVENIN_ZERO_ORDER = auto()


@dataclasses.dataclass(frozen=True)
class Hyperparameters:
    voc_model: VOCModel = VOCModel.BUSTOS_BAEZA
    ecm_model: ECMModel = ECMModel.THEVENIN_ZERO_ORDER
    ecm_model_params: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"r0": 0.1},
    )
    battery_nominal_capacity: float = 10080.0  # in Coulombs
    dt: float = 5.0  # in seconds
    v_cutoff: float = 2.5  # in Volts
    n_histories_train: int = 10
    n_histories_val: int = 5
    n_histories_test: int = 10
    process_noise_std: float = 2e-3
    measurement_noise_std: float = 5e-3
    train_seed: int = 42
    test_seed: int = 123


def gaussian_noise(*, rng: np.random.Generator, noise_std: float) -> float:
    """Draw a zero-mean Gaussian noise sample.

    Module-level (picklable) replacement for lambda-based noise distributions.
    """
    return rng.normal(loc=0.0, scale=noise_std)


def always_first_policy() -> int:
    """Deterministically select the first current policy.

    Picklable replacement for the
    default lambda of ``SimulationConfig.policy_choice_distribution``.
    """
    return 0


def bernoulli_policy_choice(*, rng: np.random.Generator, p: tuple[float, float]) -> int:
    """Stochastically select between the first two current policies.

    Picklable replacement for lambda-based policy choice.
    """
    return int(rng.choice([0, 1], p=p))


def log_simulation_config(
    config: sb.SimulationConfig, artifact_file: str = "simulation_config.pkl"
) -> None:
    """Pickle a :class:`simbat.SimulationConfig` and log it as an MLflow artifact.

    The config must be free of lambdas/closures so that it is picklable with the
    standard library ``pickle`` (all callables should be module-level functions,
    ``functools.partial`` of module-level functions, or picklable class instances).
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / artifact_file
        path.write_bytes(pickle.dumps(config))
        mlflow.log_artifact(str(path))


def load_simulation_config(
    run_id: str, artifact_file: str = "simulation_config.pkl"
) -> sb.SimulationConfig:
    """Download and unpickle a :class:`simbat.SimulationConfig` logged with
    :func:`log_simulation_config` from the given MLflow run."""
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path=artifact_file
    )
    return pickle.loads(pathlib.Path(path).read_bytes())


def _modify_dataframe(df: pd.DataFrame, run_id: int) -> None:
    """Modifies the datafram in order to result in the following schema:
    run_id, policy_id, time, load, soc, voltage"""
    df.drop(
        columns=["rul_probability", "eod_reached_sim_0"], inplace=True
    )  # Drop the RUL probability column
    df.rename(
        columns={
            "policy_id_0": "policy_id",
            "time": "time",
            "load_sim_0": "load",
            "soc_sim_0": "soc",
            "voltage_sim_0": "voltage",
        },
        inplace=True,
    )
    df.insert(0, "run_id", run_id)  # Add a run_id column for tracking


def write_histories(config: sb.SimulationConfig, n_histories: int) -> pd.DataFrame:
    """Runs MC simulations and gathers the results into a single dataframe."""
    out_df = pd.DataFrame(
        columns=["run_id", "policy_id", "time", "load", "soc", "voltage"]
    )

    for i in range(n_histories):
        result = sb.simulate_constant_capacity_simple(n_sim=1, config=config)
        df = result.to_dataframe()

        # Modify the dataframe and append it to the output one
        _modify_dataframe(df, i)

        out_df = pd.concat([out_df, df], ignore_index=True)

    return out_df


def add_rul_series(df: pd.DataFrame) -> pd.DataFrame:
    """Add a RUL series to the dataframe.

    Args:
        df: A dataframe with columns "run_id" and "time".

    Returns:
        A new dataframe with an additional column "rul" containing the remaining
        useful life for each time step.
    """
    df = df.copy()
    df["rul"] = 0.0  # Initialize the RUL column

    for run_id, group in df.groupby("run_id"):
        max_time = group["time"].max()
        df.loc[group.index, "rul"] = max_time - group["time"]

    return df


def run_discharges_from_intermediate_socs(
    config: sb.SimulationConfig,
    overrides: Iterable[Mapping[str, Any]],
    n_sim: int = 100,
) -> Iterator[sb.SimulationResult]:
    """Run Monte Carlo discharge simulations from a set of intermediate starting states.

    Args:
        config: Base simulation config (e.g. loaded from the data-generation run).
        overrides: One mapping of ``SimulationConfig`` field overrides per starting
            point, applied via :func:`dataclasses.replace` (typically ``soc_0`` and
            ``t_0``).
        n_sim: Number of Monte Carlo particles per discharge.

    Yields:
        One :class:`simbat.SimulationResult` per override.
    """
    for override in overrides:
        cfg = dataclasses.replace(config, **override)
        yield sb.simulate_constant_capacity_simple(n_sim=n_sim, config=cfg)
