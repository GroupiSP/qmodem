"""Train the backbone run of one protocol seed on the multiple-scenarios data.

The backbone run trains the mean stage, then the variance stage, and records both in
one MLflow run tagged ``role=backbone``, ``seed``, ``data_generation_run_id`` and
``stagereg_version``. It logs the mean and variance hyperparameters (flattened by
stage), every stage metric, the scalers and the finalized ``checkpoints/mean/best``
and ``checkpoints/variance/best``.

Configuration comes from the environment, falling back to ``.env``:
``RAW_DATA_DIR_MULTI``, ``DATA_GEN_RUN_ID_MULTI``, ``MLFLOW_BACKEND_STORE``,
``MLFLOW_ARTIFACT_STORE`` and ``MLFLOW_EXPERIMENT_NAME``. Variables already set in
the environment win over ``.env``, so a run can be pointed at another backend.

Protocol backbones (seed 0 included) are re-trained from the frozen HPs of the mean
and variance studies, passed as ``--mean-study-run-id`` and
``--variance-study-run-id`` (their ``best.*`` params); epoch caps and patience
stay as configured here. Without them, the defaults are used. The seed-0 backbone
the variance study runs on takes ``--mean-study-run-id`` alone (default variance LR).

Example:
    uv run scripts/battery_multiple_scenarios/train_backbone.py --seed 0
    uv run scripts/battery_multiple_scenarios/train_backbone.py --seed 0 \
        --mean-study-run-id <ID>
    uv run scripts/battery_multiple_scenarios/train_backbone.py --seed 1 \
        --mean-study-run-id <ID> --variance-study-run-id <ID>
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
from collections.abc import Sequence

import mlflow
import numpy as np
import sklearn.preprocessing as skpp
from dotenv import load_dotenv

from qmodem.battery.data_processing import (
    FEATURES,
    WINDOW_SIZE,
    make_data_pipeline,
    prepare_data,
)
from qmodem.battery.hpo import frozen_backbone_hyperparameters
from qmodem.battery.train import (
    MeanHyperparameters,
    StageScheduleHyperparameters,
    VarianceHyperparameters,
    track_backbone_run,
)
from qmodem.tracking import MLFlowSetup, MLflowTracker, ScalerArtifact
from qmodem.utils import setup_script_logging

logger = logging.getLogger(__name__)

_DEFAULT_SCHEDULE = StageScheduleHyperparameters()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="Protocol seed r.")
    parser.add_argument(
        "--mean-epochs",
        type=int,
        default=_DEFAULT_SCHEDULE.max_epochs,
        help="Epoch cap of the mean stage.",
    )
    parser.add_argument(
        "--variance-epochs",
        type=int,
        default=_DEFAULT_SCHEDULE.max_epochs,
        help="Epoch cap of the variance stage.",
    )
    parser.add_argument(
        "--mean-study-run-id",
        default=None,
        help="Mean study run whose frozen best.* params set the mean HPs.",
    )
    parser.add_argument(
        "--variance-study-run-id",
        default=None,
        help=(
            "Variance study run whose frozen best.* params set the variance LR "
            "(omit it for the seed-0 backbone the variance study runs on)."
        ),
    )
    parser.add_argument("--run-name", default=None, help="MLflow run name.")
    args = parser.parse_args(argv)
    if args.variance_study_run_id is not None and args.mean_study_run_id is None:
        parser.error("--variance-study-run-id needs --mean-study-run-id.")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Train and record one backbone run.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    load_dotenv()
    log_stream = setup_script_logging()

    setup = MLFlowSetup(run_name=args.run_name or f"backbone-seed-{args.seed}")
    data_generation_run_id = os.environ["DATA_GEN_RUN_ID_MULTI"]
    mlflow.set_tracking_uri(setup.backend_store)  # prepare_data reads the data run

    x_scaler = skpp.StandardScaler()
    y_scaler = skpp.MinMaxScaler(feature_range=(0, 1))
    data = prepare_data(
        pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
        data_generation_run_id,
        make_data_pipeline(WINDOW_SIZE, x_scaler=x_scaler, y_scaler=y_scaler),
    )
    logger.info("Windows: %d train, %d validation", len(data.train), len(data.val))

    mean_hp = MeanHyperparameters(
        schedule=StageScheduleHyperparameters(max_epochs=args.mean_epochs)
    )
    variance_hp = VarianceHyperparameters(
        schedule=StageScheduleHyperparameters(max_epochs=args.variance_epochs)
    )
    if args.mean_study_run_id is not None:
        mean_hp, variance_hp = frozen_backbone_hyperparameters(
            mean_study_run_id=args.mean_study_run_id,
            variance_study_run_id=args.variance_study_run_id,
            mean_base=mean_hp,
            variance_base=variance_hp,
        )
        logger.info("Frozen backbone HPs: %s, %s", mean_hp, variance_hp)
    run = track_backbone_run(
        data=data,
        mean_hp=mean_hp,
        variance_hp=variance_hp,
        seed=args.seed,
        data_generation_run_id=data_generation_run_id,
        scalers=(
            ScalerArtifact("x_scaler", x_scaler, np.ones((WINDOW_SIZE, len(FEATURES)))),
            ScalerArtifact("y_scaler", y_scaler, np.ones((1, 1)), "inverse_transform"),
        ),
        setup=setup,
    )
    logger.info(
        "Backbone run %s finalized mean epoch %d and variance epoch %d",
        run.run_id,
        run.checkpoints.mean.epoch,
        run.checkpoints.variance.epoch,
    )
    MLflowTracker(run.run_id).log_artifact("training_log.txt", log_stream.getvalue())


if __name__ == "__main__":
    main()
