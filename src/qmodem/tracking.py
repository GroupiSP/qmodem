from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Generator

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
    run_name: str
    experiment_name: str
    tags: Tags
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
        active_run = mlflow.start_run(run_name=setup.run_name)
        mlflow.set_tags(asdict(setup.tags))

        yield active_run

    finally:
        mlflow.end_run()


# TODO: implement
def get_tags_from_mlflow_run(run_id: str) -> dict[str, str]:
    pass
