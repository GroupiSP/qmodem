from __future__ import annotations

import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Any, Generator

import mlflow
import orbax.checkpoint as ocp

from .train import TrainingContext, TrainingPhase
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
    run_id: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)
    backend_store: str = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
    artifact_store: str | Path = ROOT_DIR / "mlruns"
    tracking_server: str | None = None
    """Configuration for an MLflow tracking run.

    Attributes:
        run_name: Human-readable name for the MLflow run.
        experiment_name: Name of the MLflow experiment to log under.
        run_id: Optional existing run ID to resume. If None, a new run is created.
        tags: Arbitrary key-value tags attached to the run.
        backend_store: SQLAlchemy URI for the MLflow backend store.
        artifact_store: Local path where artifacts are stored.
        tracking_server: Remote tracking server URI (not yet supported).
    """

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


def mlflow_track_model_best_state(
    phase: TrainingPhase, context: TrainingContext
) -> None:
    if phase == TrainingPhase.BEFORE_RETURN:
        run = mlflow.active_run()
        if run is None:
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "best_model_state"
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(ckpt_path, context.model_best_state)
            time.sleep(0.1)  # let Orbax finish async writes
            mlflow.log_artifacts(str(ckpt_path), artifact_path="best_model_state")


def mlflow_track_losses(phase: TrainingPhase, context: TrainingContext) -> None:
    if phase == TrainingPhase.EPOCH_END:
        mlflow.log_metric("train_loss", context.train_loss, step=context.epoch)
        mlflow.log_metric("val_loss", context.val_loss, step=context.epoch)
        mlflow.log_metric("best_val_loss", context.best_val_loss, step=context.epoch)


# TODO: implement
def get_tags_from_mlflow_run(run_id: str) -> dict[str, str]:
    pass
