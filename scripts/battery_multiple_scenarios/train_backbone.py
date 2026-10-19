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
the variance study runs on takes ``--mean-study-run-id`` alone (default variance LR);
it is then re-trained with both studies before the posterior studies, which refuse a
backbone without a ``variance_study_run_id`` tag. The study run IDs are recorded as
the tags ``mean_study_run_id`` / ``variance_study_run_id``.

Epoch caps other than the protocol's (``--mean-epochs`` / ``--variance-epochs``) tag
the run ``protocol_overrides=true`` (with ``protocol_overrides.params``), so tag
selection skips it unless ``--include-overridden`` is passed.

Example:
    uv run scripts/battery_multiple_scenarios/train_backbone.py --seed 0
    uv run scripts/battery_multiple_scenarios/train_backbone.py --seed 0 \
        --mean-study-run-id <ID>
    uv run scripts/battery_multiple_scenarios/train_backbone.py --seed 1 \
        --mean-study-run-id <ID> --variance-study-run-id <ID>
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
from collections.abc import Sequence

import numpy as np
import sklearn.preprocessing as skpp

from qmodem.battery.data_processing import (
    FEATURES,
    WINDOW_SIZE,
    log_window_counts,
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
from qmodem.script_session import script_session
from qmodem.tracking import (
    ScalerArtifact,
    Stage,
    protocol_override_tags,
)

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


def backbone_hyperparameters(
    args: argparse.Namespace,
) -> tuple[MeanHyperparameters, VarianceHyperparameters]:
    """Return the backbone's HPs: the CLI caps, then the studies' frozen HPs.

    Args:
        args: The parsed arguments.

    Returns:
        The mean and variance hyperparameters.
    """
    mean_hp = MeanHyperparameters(
        schedule=StageScheduleHyperparameters(max_epochs=args.mean_epochs)
    )
    variance_hp = VarianceHyperparameters(
        schedule=StageScheduleHyperparameters(max_epochs=args.variance_epochs)
    )
    if args.mean_study_run_id is None:
        return mean_hp, variance_hp
    mean_hp, variance_hp = frozen_backbone_hyperparameters(
        mean_study_run_id=args.mean_study_run_id,
        variance_study_run_id=args.variance_study_run_id,
        mean_base=mean_hp,
        variance_base=variance_hp,
    )
    logger.info("Frozen backbone HPs: %s, %s", mean_hp, variance_hp)
    return mean_hp, variance_hp


def override_tags(args: argparse.Namespace) -> dict[str, str]:
    """Return the ``protocol_overrides`` tags of non-protocol epoch caps.

    Args:
        args: The parsed arguments.

    Returns:
        The tags (empty when both caps are the protocol's).
    """
    return protocol_override_tags(
        [
            f"{stage}/schedule/max_epochs"
            for stage, epochs in (
                (Stage.MEAN, args.mean_epochs),
                (Stage.VARIANCE, args.variance_epochs),
            )
            if epochs != _DEFAULT_SCHEDULE.max_epochs
        ]
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Train and record one backbone run.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    with script_session() as session:
        data_generation_run_id = os.environ["DATA_GEN_RUN_ID_MULTI"]
        x_scaler = skpp.StandardScaler()
        y_scaler = skpp.MinMaxScaler(feature_range=(0, 1))
        data = prepare_data(
            pathlib.Path(os.environ["RAW_DATA_DIR_MULTI"]),
            data_generation_run_id,
            make_data_pipeline(WINDOW_SIZE, x_scaler=x_scaler, y_scaler=y_scaler),
        )
        logger.info("Windows per split: %s", data.window_counts())
        mean_hp, variance_hp = backbone_hyperparameters(args)
        run = track_backbone_run(
            data=data,
            mean_hp=mean_hp,
            variance_hp=variance_hp,
            seed=args.seed,
            data_generation_run_id=data_generation_run_id,
            scalers=(
                ScalerArtifact(
                    "x_scaler", x_scaler, np.ones((WINDOW_SIZE, len(FEATURES)))
                ),
                ScalerArtifact(
                    "y_scaler", y_scaler, np.ones((1, 1)), "inverse_transform"
                ),
            ),
            setup=dataclasses.replace(
                session.setup,
                run_name=args.run_name or f"backbone-seed-{args.seed}",
                tags=override_tags(args),
            ),
            mean_study_run_id=args.mean_study_run_id,
            variance_study_run_id=args.variance_study_run_id,
        )
        logger.info(
            "Backbone run %s finalized mean epoch %d and variance epoch %d",
            run.run_id,
            run.checkpoints.mean.epoch,
            run.checkpoints.variance.epoch,
        )
        log_window_counts(run.run_id, data, client=session.client)
        session.attach_log(run.run_id, "training_log.txt")


if __name__ == "__main__":
    main()
