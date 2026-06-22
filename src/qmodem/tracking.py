from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Generator

import mlflow

from .utils import ROOT_DIR


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
class Tags:
    dataset: DatasetChoice = DatasetChoice.CMAPSS
    loss: LossChoice = LossChoice.MSE
    model: ModelChoice = ModelChoice.LSTM
    optimizer: OptimizerChoice = OptimizerChoice.ADAM
    scheduler: SchedulerChoice = SchedulerChoice.COSINE
    hp_sampler: HPSamplerChoice = HPSamplerChoice.RANDOM
    hp_pruner: HPPrunerChoice = HPPrunerChoice.MEDIAN


@dataclass(frozen=True)
class MLFlowSetup:
    """Configuration for an MLflow tracking run.

    Attributes:
        experiment_name: Name of the MLflow experiment to log under.
        run_name: Optional human-readable name for the MLflow run. If None, MLflow will auto-generate a name.
        run_id: Optional existing run ID to resume. If None, a new run is created.
        tags: Arbitrary key-value tags attached to the run.
        backend_store: SQLAlchemy URI for the MLflow backend store.
        artifact_store: Local path where artifacts are stored.
        tracking_server: Remote tracking server URI (not yet supported).
    """

    experiment_name: str
    run_name: str | None = None
    run_id: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    backend_store: str = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
    artifact_store: str | Path = ROOT_DIR / "mlruns"
    tracking_server: str | None = None

    def __post_init__(self):
        if self.tracking_server is not None:
            raise NotImplementedError("Remote tracking server is not supported yet.")


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
        active_run = mlflow.start_run(run_id=setup.run_id, run_name=setup.run_name)
        mlflow.set_tags(setup.tags)

        yield active_run

    finally:
        mlflow.end_run()


# TODO: implement
def get_tags_from_mlflow_run(run_id: str) -> dict[str, str]:
    pass
