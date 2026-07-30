from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Generator

import mlflow
import pandas as pd

from .utils import LAST_TRAIN_SETUP_PATH, ROOT_DIR


class DatasetChoice(StrEnum):
    BATTERY = auto()
    CMAPSS = auto()


class LossChoice(StrEnum):
    MSE = auto()
    NLL = auto()


class ModelChoice(StrEnum):
    CNN = auto()
    LSTM = auto()


class OptimizerChoice(StrEnum):
    ADAM = auto()
    SGD = auto()


class SchedulerChoice(StrEnum):
    COSINE = auto()
    STEP = auto()


class HPSamplerChoice(StrEnum):
    RANDOM = auto()
    TPE = auto()


class HPPrunerChoice(StrEnum):
    ASHA = auto()
    MEDIAN = auto()
    NOP = auto()


@dataclass(frozen=True)
class MLFlowSetup:
    """Configuration for an MLflow tracking run.

    Attributes:
        experiment_name: Name of the MLflow experiment to log under.
        run_name: Optional human-readable name for the MLflow run. If None, MLflow will auto-generate a name.
        run_id: Optional existing run ID to resume. If None, a new run is created.
        run_description: Optional description for the MLflow run.
        run_nested: Whether this run is a nested run. Defaults to False.
        tags: Arbitrary key-value tags attached to the run.
        backend_store: SQLAlchemy URI for the MLflow backend store. Defaults to the value of the `MLFLOW_BACKEND_STORE` environment variable or `sqlite:///mlflow.db` if not set.
        artifact_store: Local path where artifacts are stored.
        tracking_server: Remote tracking server URI (not yet supported).
    """

    experiment_name: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    run_description: str | None = None
    run_nested: bool = False
    tags: dict[str, Any] = field(default_factory=dict)
    backend_store: str | None = None
    artifact_store: str | None = None
    tracking_server: str | None = None

    def _are_all_defaults_nested_true(self) -> bool:
        """Checks if all attributes except `run_nested` are set to their default
        values."""
        return (
            self.experiment_name is None
            and self.run_name is None
            and self.run_id is None
            and self.run_description is None
            and self.backend_store is None
            and self.artifact_store is None
            and self.tracking_server is None
        )

    def __post_init__(self):
        if self.tracking_server is not None:
            raise NotImplementedError("Remote tracking server is not supported yet.")
        if self.run_nested and not self._are_all_defaults_nested_true():
            raise ValueError(
                "When run_nested is True, all other arguments must be defaults."
            )
        if self.experiment_name is None:
            object.__setattr__(
                self,
                "experiment_name",
                os.environ.get("MLFLOW_EXPERIMENT_NAME", "Default"),
            )
        if self.backend_store is None:
            object.__setattr__(
                self,
                "backend_store",
                os.environ.get("MLFLOW_BACKEND_STORE", "sqlite:///mlflow.db"),
            )
        if self.artifact_store is None:
            object.__setattr__(
                self,
                "artifact_store",
                os.environ.get("MLFLOW_ARTIFACT_STORE", ROOT_DIR / "mlruns"),
            )


@contextmanager
def track_mlflow(setup: MLFlowSetup) -> Generator[mlflow.ActiveRun, None, None]:
    mlflow.set_tracking_uri(setup.backend_store)

    exp_name = mlflow.get_experiment_by_name(setup.experiment_name)
    exp_id = (
        exp_name.experiment_id
        if exp_name is not None
        else mlflow.create_experiment(
            setup.experiment_name, artifact_location=str(setup.artifact_store)
        )
    )
    mlflow.set_experiment(experiment_id=exp_id)

    try:
        active_run = mlflow.start_run(
            run_id=setup.run_id,
            run_name=setup.run_name,
            nested=setup.run_nested,
            description=setup.run_description,
        )
        mlflow.set_tags(setup.tags)

        yield active_run

    finally:
        mlflow.end_run()


# TODO: implement
def get_tags_from_mlflow_run(run_id: str) -> dict[str, str]:
    pass


def track_dataframe(df: pd.DataFrame, name: str, context: str) -> None:
    dataset = mlflow.data.from_pandas(df, name=name)
    mlflow.log_input(dataset=dataset, context=context)


def write_setup_to_file(experiment_name: str) -> None:
    """Writes only the setup information necessary to resume a trained run for testing
    purposes."""
    active_run = mlflow.active_run()

    with open(LAST_TRAIN_SETUP_PATH, "w") as f:
        json.dump(
            {
                "run_id": active_run.info.run_id,
                "experiment_name": experiment_name,
            },
            f,
        )

    return


def retrieve_mlflow_setup_train() -> MLFlowSetup:
    """Retrieves the MLFlowSetup from a JSON file at the given path. If the file does
    not exist, prompts the user for input.

    Returns:
        An instance of MLFlowSetup.
    """
    use_last = os.environ.get("MLFLOW_USE_LAST_TRAINED", "").lower() in ("1", "true")
    if use_last:
        with open(LAST_TRAIN_SETUP_PATH, "r") as f:
            data = json.load(f)
        return MLFlowSetup(
            run_id=data["run_id"],
            experiment_name=data["experiment_name"],
        )
    else:
        print(
            "MLFLOW_USE_LAST_TRAINED is set to False. Please input the training run ID and experiment name."
        )
        run_id = input("Enter the training run ID: ").strip()
        experiment_name = input("Enter the experiment name: ").strip()
        return MLFlowSetup(
            run_id=run_id,
            experiment_name=experiment_name,
        )


def get_run_parameters(run_id: str, backend_store: str) -> dict[str, Any]:
    """Retrieves the parameters of a specific MLflow run.

    Args:
        run_id (str): The ID of the MLflow run.

    Returns:
        dict[str, Any]: A dictionary containing the parameters of the run.
    """
    mlflow.set_tracking_uri(backend_store)
    run = mlflow.get_run(run_id)
    return run.data.params
