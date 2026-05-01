from dataclasses import asdict, dataclass
from enum import StrEnum, auto
from pathlib import Path

import mlflow

from .utils import ROOT_DIR


class DatasetChoice(StrEnum):
    BATTERY = auto()
    CMAPSS = auto()


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
    model: ModelChoice = ModelChoice.LSTM
    dataset: DatasetChoice = DatasetChoice.CMAPSS
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


# TODO: turn into a context manager that also ends the run at the end of the context
def setup_mlflow_tracking(setup: MLFlowSetup) -> None:
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

    mlflow.start_run(run_name=setup.run_name)
    if setup.tags:
        mlflow.set_tags(asdict(setup.tags))


# TODO: implement
def get_tags_from_mlflow_run(run_id: str) -> dict[str, str]:
    pass
